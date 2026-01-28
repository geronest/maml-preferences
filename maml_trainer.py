import torch
import torch.nn as nn
import random
import gc

from torch.nn import functional as F

from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_peft_available
if is_peft_available():
    from peft import (
        PeftModel,
    )

from datasets import  DatasetDict
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader
from torch.func import functional_call
from torch import optim
from typing import Any, Callable, Optional, Union

from copy import deepcopy

from trl.trainer.utils import compute_accuracy
from trl.models.modeling_base import create_reference_model
from trl import DPOTrainer
from transformers.trainer_utils import EvalLoopOutput

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def maml_collate_fn(batch):
    return batch[0]

def compute_loss_sft(model, inputs, params = None, return_outputs=False, num_items_in_batch=None, **kwargs):
    if params is not None:
        outputs = functional_call(model, params, (), kwargs=inputs)
    else:
        outputs = model(**inputs)

    ##print("after computing outputs for sft loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    ##print("after get sft inner loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    return loss

def compute_loss_reward(model, inputs, params = None, return_outputs=False, num_items_in_batch=None, **kwargs):
    if params is not None:
        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"],
            "attention_mask": inputs["attention_mask_chosen"],
            "return_dict": True
        }
        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"],
            "attention_mask": inputs["attention_mask_rejected"],
            "return_dict": True
        }
        rewards_chosen = functional_call(
            model, 
            params,
            (),
            kwargs=inputs_chosen
        )["logits"].squeeze(-1)

        rewards_rejected = functional_call(
            model, 
            params,
            (),
            kwargs=inputs_rejected
        )["logits"].squeeze(-1)

    else:
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"].squeeze(-1)

        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"].squeeze(-1)

    loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
    # print("Grad fn for reward_chosen:", rewards_chosen.grad_fn)

    if return_outputs:
        return loss, {
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
        }
    return loss

def compute_loss_dpo(
    model, 
    inputs,
    params = None, 
    return_outputs=False, 
    num_items_in_batch=None, 
    ref_chosen=None, 
    ref_rejected=None,
    beta=0.1,
    pad_token_id=None, 
    **kwargs
):
    # print("Before computing policy output: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    policy_output = functional_forward_pass(model, inputs, pad_token_id, params)
    # print("after computing policy output: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    policy_chosen = policy_output["chosen_logps"]
    policy_rejected = policy_output["rejected_logps"]

    loss, metrics = calculate_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta
    )
    # print("after computing dpo loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    if return_outputs:
        return loss, metrics
    return loss

def functional_forward_pass(
    model,
    batch,
    pad_token_id,
    params
):
    original_batch_size = batch["chosen_input_ids"].shape[0]
    # Prepare inputs
    concatenated_batch = DPOTrainer.concatenated_inputs(batch, padding_value=pad_token_id)
    
    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]
    
    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
        dim=1,
    )

    model_kwargs = {"use_cache": False}
    model_kwargs["attention_mask"] = attention_mask

    # Generate outputs
    # print("Before policy generating output: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    if params is not None:
        model_kwargs["input_ids"] = input_ids
        outputs = functional_call(model, params, (), kwargs=model_kwargs)
    else:
        outputs = model(input_ids, **model_kwargs)
    # print("After policy generating output: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    logits = outputs.logits
    del outputs
    torch.cuda.empty_cache()
    # print("Before compute logps: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    logps = compute_logps(logits, input_ids, loss_mask, original_batch_size)
    
    return logps

def compute_logps(logits, input_ids, loss_mask, original_batch_size):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_loss_mask = loss_mask[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1, dtype=shift_logits.dtype)
    per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    masked_logps = per_token_logps * shift_loss_mask
    all_logps = masked_logps.sum(dim=-1)

    chosen_logps = all_logps[:original_batch_size]
    rejected_logps = all_logps[original_batch_size:]

    return {
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
    }

def calculate_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta = 0.1,
    label_smoothing = 0.0,
):

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    # Sigmoid   
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    
    loss = losses.mean()

    # Create metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps.detach() - ref_chosen_logps.detach())
        rejected_rewards = beta * (policy_rejected_logps.detach() - ref_rejected_logps.detach())
        
        # reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # reward_margins = chosen_rewards - rejected_rewards
        
        metrics = {
            "rewards/chosen": chosen_rewards,
            "rewards/rejected": rejected_rewards,
            # "rewards/accuracies": reward_accuracies.mean().item(),
            # "rewards/margins": reward_margins.mean().item(),
            # "logps/chosen": policy_chosen_logps.detach().mean().item(),
            # "logps/rejected": policy_rejected_logps.detach().mean().item(),
            # "logits": logits.detach().mean().item(),
        }
        
    return loss, metrics

class MAMLTrainer(Trainer):
    def __init__(
        self, 
        model, 
        args, 
        train_dataset, 
        inner_train_batch_size, 
        inner_lr_coef=0.1, 
        inner_optimizer='SGD', 
        model_type="sft",
        first_order=True,
        ref_model=None,
        num_inner_steps=1,
        low_memory_mode=False,
        **kwargs
    ):
        # self.inner_lr = inner_lr
        self.inner_lr_coef = inner_lr_coef
        self.inner_train_batch_size = inner_train_batch_size
        self.inner_optimizer = inner_optimizer
        self.num_inner_steps = num_inner_steps
        self.current_tasks = None
        self.model_type = model_type
        self.first_order=first_order
        self.ref_model = ref_model
        # self.step = 0
        
        func_compute_metrics = None
        # if model_type == "reward":
        #     func_compute_metrics = compute_accuracy
        #     #print("\n#####\nUsing compute_accuracy for evaluation!\n#####\n")
        self.beta = args.beta if self.model_type=='dpo' else None

        if self.model_type=='dpo':
            self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model or args.precompute_ref_log_probs:
                # The `model` with adapters turned off will be used as the reference model
                self.ref_model = None
            else:
                self.ref_model = create_reference_model(model)


        super().__init__(
            model = model,
            args = args,
            train_dataset = train_dataset,
            compute_metrics=func_compute_metrics,
            **kwargs
        )
        pad_token = self.processing_class.pad_token or self.processing_class.eos_token
        self.pad_token_id = self.processing_class.convert_tokens_to_ids(pad_token)

    def reset_model_params(self, model, params):
        for n, p in model.named_parameters():
            if p.requires_grad and n in params:
                p.data.copy_(params[n].data)

    # 'training_step' returns loss to optimize the model parameters Phi
    def training_step(
        self, model, inputs, num_items_in_batch=None, is_evaluate=False
    ):
        # print("Start training step: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

        model.train()

        inputs = self._prepare_inputs(inputs)
        # print("After prepare inputs: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        
        # self.step+=1
        # ##print(f"Current step: {self.step}")
        
        shared_loss = torch.tensor(0.0, device=self.args.device)
        
        accumulated_grads = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        # print("after initialising accum-grads: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        for language, batch in inputs.items():

            # Split the batch into inner and outer loop data
            batch_inner = {}
            batch_outer = {}
            for key, value in batch.items():
                split_point = value.size(0) // (1 + self.num_inner_steps)
                batch_inner[key] = value[split_point:]
                batch_outer[key] = value[:split_point]

            # for key, value in batch_inner.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"inner Key: '{key}', Shape: {value.shape}")

            # print("Before adapt param: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            inner_updated_params = self.adapt_params(
                # model_temp, 
                model, 
                batch_inner, 
                is_evaluate
            )
            # print("After adapt param: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            outer_ref_chosen = None
            outer_ref_rejected = None
            if self.model_type == "dpo":
                outer_ref_chosen, outer_ref_rejected = self.compute_ref_output(batch_outer)

            # print("after compute outer ref output: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

            outer_loss = self.compute_loss(
                model, 
                batch_outer,
                params=inner_updated_params,
                ref_chosen=outer_ref_chosen,
                ref_rejected=outer_ref_rejected
            )
            # print("After computing loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

            del batch_outer
            
            if not is_evaluate:
                # Compute gradients of outer loss w.r.t. original model parameters
                # This is the key insight of MAML: we need gradients w.r.t. θ, not θ'
                # print("Before meta_grads: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                if self.first_order:
                    meta_grads = torch.autograd.grad(
                        outer_loss, # L(θ')
                        tuple(inner_updated_params.values()), # θ'
                        retain_graph=False,
                        allow_unused=True
                    )
                    # print("after meta-grads: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                    for (name, _), g in zip(inner_updated_params.items(), meta_grads):
                        if g is not None and name in accumulated_grads:
                            accumulated_grads[name].add_(g.detach(), alpha=1.0 / len(inputs))
                else:
                    meta_grads = torch.autograd.grad(
                        outer_loss, # L(θ')
                        [p for p in model.parameters() if p.requires_grad], # θ
                        retain_graph=False,
                        allow_unused=True
                    )
                
                # Accumulate gradients
                    for (name, param), grad in zip(
                        [(n, p) for n, p in model.named_parameters() if p.requires_grad], 
                        meta_grads
                    ):
                        if grad is not None and name in accumulated_grads:
                            accumulated_grads[name].add_(grad.detach(), alpha=1.0 / len(inputs))
                del meta_grads
            del outer_ref_chosen, outer_ref_rejected

            shared_loss += outer_loss/len(inputs)
            #print("after updating shared loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

            # del outer_loss, model_temp
            # del outer_loss, inner_updated_params
            del outer_loss
            torch.cuda.empty_cache()

            # Force garbage collection every few iterations to prevent memory leaks
            if len(inputs) > 2:  # Only for multi-task batches
                gc.collect()
        
        # theta' = theta + beta * mean(dL(phi')/dphi')
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = accumulated_grads[name]

        del accumulated_grads
        torch.cuda.empty_cache()

        del inputs

        torch.cuda.empty_cache()
        gc.collect()
        # print("Finish one training step: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

        return shared_loss.detach()

    def compute_loss(self, model, inputs, params = None, return_outputs=False, num_items_in_batch=None, ref_chosen=None, ref_rejected=None):

        if self.model_type == "sft":
            func_compute_loss = compute_loss_sft
        elif self.model_type == "reward":
            func_compute_loss = compute_loss_reward
        elif self.model_type == "dpo":
            func_compute_loss = compute_loss_dpo

        with self.compute_loss_context_manager():
            loss = func_compute_loss(
                model,
                inputs,
                params,
                return_outputs,
                num_items_in_batch,
                ref_model=self.ref_model, 
                beta=self.beta,
                pad_token_id=self.pad_token_id, 
                ref_chosen=ref_chosen,
                ref_rejected=ref_rejected
            )

        return loss

    def set_param_grads(self, model, dict_params):
        for name, param in model.named_parameters():
            if param.requires_grad:
                dict_params[name].grad = param.grad

    def reset_model_grads(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = None

    def initialize_inner_optimizer(self, params):
        optimizers = {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
        }

        if self.inner_optimizer not in optimizers:
            raise ValueError(f"Unsupported optimizer type: {self.inner_optimizer}")
        
        # return optimizers[self.inner_optimizer](params.values(), lr=self.inner_lr)
        return optimizers[self.inner_optimizer](params, lr=self.inner_lr)

    def compute_ref_output(self, sampled_data, eval_mode=False):
        with torch.no_grad():            
            if self.ref_model is None: # PEFT case
                with self.model.disable_adapter():
                    self.model.eval()
                    ref_output = functional_forward_pass(self.model, sampled_data, self.pad_token_id, params=None)
                    if not eval_mode:
                        self.model.train()
            else: # Standard case
                self.ref_model.eval()
                ref_output = functional_forward_pass(self.ref_model, sampled_data, self.pad_token_id, params=None)

        ref_chosen = ref_output["chosen_logps"]
        ref_rejected = ref_output["rejected_logps"]

        return ref_chosen, ref_rejected


    def adapt_params(self, model, sampled_data, is_evaluate = False):
        # Create copies of parameters that maintain computational graph
        if self.first_order:
            adapted_params = {
                n: p.detach().clone().requires_grad_() for n, p in model.named_parameters() if p.requires_grad
            }
        else:
            adapted_params = {
                n: p.clone() for n, p in model.named_parameters() if p.requires_grad
            }

        # Now utilize self.num_inner_steps to perform multiple inner updates (multi-step adaptation)
        # We'll apply self.num_inner_steps inner loop updates, each time using the latest updated params.
        # If self.num_inner_steps is 1 (the default), this does one step.
        current_params = adapted_params
        items_per_step = len(sampled_data[list(sampled_data.keys())[0]]) // self.num_inner_steps if self.num_inner_steps > 0 else 0
        for i in range(self.num_inner_steps):
            i_sampled_data = {k: v[i*items_per_step:(i+1)*items_per_step] for k, v in sampled_data.items()}
            ref_chosen = None
            ref_rejected = None
            if self.model_type == "dpo":
                ref_chosen, ref_rejected = self.compute_ref_output(i_sampled_data)
            inner_loss = self.compute_loss(
                model, 
                i_sampled_data, 
                params=current_params,
                ref_chosen=ref_chosen,
                ref_rejected=ref_rejected
            )
            inner_grads = torch.autograd.grad(
                inner_loss,
                current_params.values(),
                create_graph=(not self.first_order and not is_evaluate),
                retain_graph=(not self.first_order),
                allow_unused=True
            )
            updated_params = {}
            lr_inner = self.inner_lr_coef * self._get_learning_rate()
            for (name, param), grad in zip(current_params.items(), inner_grads):
                if grad is not None:
                    updated_params[name] = param - lr_inner * grad
            # Prepare for next step
            del inner_grads
            del inner_loss
            del ref_chosen, ref_rejected
            torch.cuda.empty_cache()
            gc.collect()
            current_params = {name: updated_params.get(name, param) for name, param in current_params.items()}
        # Replace single step below with multi-step adaptation
        adapted_params = current_params
        # ref_chosen = None
        # ref_rejected = None
        # if self.model_type == "dpo":
        #     ref_chosen, ref_rejected = self.compute_ref_output(sampled_data)

        # # print("Before computing inner loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        
        # inner_loss = self.compute_loss(
        #     model, 
        #     sampled_data, 
        #     params=adapted_params,
        #     ref_chosen=ref_chosen,
        #     ref_rejected=ref_rejected
        # )
        # # print("After computing inner loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        
        # inner_grads = torch.autograd.grad(
        #     inner_loss,
        #     adapted_params.values(),
        #     create_graph=(not self.first_order and not is_evaluate), # Not create graph if using FOMAML
        #     retain_graph=(not self.first_order),
        #     allow_unused=True
        # )
        # # print("After computing inner grads: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        # # print("Any requires_grad in adapted_params:", any(p.requires_grad for p in adapted_params.values()))
        # # print("-----inner grads------")
        # # print(inner_grads)
        # # Manual gradient step: θ' = θ - α∇_θ L_inner
        # updated_params = {}
        # for (name, param), grad in zip(adapted_params.items(), inner_grads):
        #     lr_inner = self.inner_lr_coef * self._get_learning_rate()
        #     if grad is not None:
        #         # new_param = param - lr_inner * grad
        #         # if self.first_order:
        #         #     new_param = new_param.detach().requires_grad_()
        #         updated_params[name] = param - lr_inner * grad
        # # print("After updating inner params: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

        # # Free inner_grads immediately after use
        # del inner_grads
        # del ref_chosen, ref_rejected
    
        # # Clean up remaining tensors
        # del inner_loss, adapted_params
        torch.cuda.empty_cache()
        gc.collect()  # Additional cleanup for adapt_params

        # return updated_params
        return adapted_params

    def get_train_dataloader(self):
        """
            As meta learning consists of inner and outer loop, sample only languages 
            from train_dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=1, # Each item from MAMLDataset is a dict of batches
            collate_fn=maml_collate_fn,
            # Ensure num_workers and pin_memory are set appropriately in TrainingArguments
            # num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )

        # No need to set self.train_dataloader here, Trainer does this
        return train_dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1, # Each item from MAMLDataset is a dict of batches
            collate_fn=maml_collate_fn,
            # Ensure num_workers and pin_memory are set appropriately
            # num_workers=self.args.dataloader_num_workers,
            # pin_memory=self.args.dataloader_pin_memory,
        )
         # No need to set self.eval_dataloader here, Trainer does this
        return eval_dataloader

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # Use no_grad context for evaluation!
        with self.compute_loss_context_manager():
            
            task_losses = [] # Store individual task losses temporarily
            task_logits = []
            for language, batch in inputs.items():
                # Ensure batch is prepared (e.g., on device) if not already
                prepared_batch = self._prepare_inputs(batch)
                # for key, value in prepared_batch.items():
                #     if isinstance(value, torch.Tensor):
                #         print(f"Key: '{key}', Shape: {value.shape}")

                # Split the batch into inner and outer loop data
                # batch_inner = {}
                # batch_outer = {}
                # for key, value in prepared_batch.items():
                #     split_point = value.size(0) // 2
                    # batch_inner[key] = value[:split_point]
                    # batch_outer[key] = value[split_point:]

                # print("Before eval compute loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                
                with torch.no_grad():
                    logits_language = None
                    if self.model_type in ["reward"]:
                        loss_language, logits_language = self.compute_loss(
                            model,
                            prepared_batch,
                            params=None,
                            num_items_in_batch=None, # This arg seems unused in compute_loss
                            return_outputs = True
                        )
                    elif self.model_type == "sft":
                        loss_language = self.compute_loss(
                            model,
                            prepared_batch,
                            params=None,
                            num_items_in_batch=None, # This arg seems unused in compute_loss
                            return_outputs = True
                        )
                    elif self.model_type == "dpo":
                        ref_chosen, ref_rejected = self.compute_ref_output(prepared_batch, eval_mode=True)
                        torch.cuda.empty_cache()
                        # print("After eval ref output: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                        loss_language, logits_language = self.compute_loss(
                            model,
                            prepared_batch,
                            params=None,
                            num_items_in_batch=None, # This arg seems unused in compute_loss
                            return_outputs = True,
                            ref_chosen=ref_chosen,
                            ref_rejected=ref_rejected
                        )
                        del ref_chosen, ref_rejected
                
                # print("After eval compute loss: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

                    
                task_losses.append(loss_language)
                task_logits.append(logits_language)
                # print("logits_language: ", logits_language)

                # Explicitly delete intermediate tensors within the loop if memory is tight
                # del inner_updated_params
                del prepared_batch

        # Average the losses after the loop
        if task_losses:
            loss = torch.stack(task_losses).mean()

        # Clean up list and input dict reference
        del task_losses
        del inputs

        # Logits and labels are usually returned for metric computation,
        # but MAML evaluation might only need loss.
        # Returning None for logits and labels if only loss is needed.
        # Ensure loss is detached and on CPU for aggregation if needed by Trainer internals.
        loss = loss.detach().mean() # Average loss over the batch if it's not already scalar

        if self.model_type in ["reward", "dpo"]:
            logits = {
                k: torch.concat([item[k] for item in task_logits], axis=0) for k in task_logits[0]
            }
            logits = tuple(v for k, v in logits.items())
            logits = nested_detach(logits)
            # Stack accepted against rejected, mean over logits
            # and softmax to get preferences between accepted and rejected to sum to 1
            logits = torch.stack(logits)
            if len(logits.shape) == 3:
                logits = logits.mean(dim=2).softmax(dim=0).T
            elif len(logits.shape) == 2:
                logits = logits.softmax(dim=0).T
    
            labels = torch.zeros(logits.shape[0])
            labels = self._prepare_inputs(labels)
            
            del task_logits


            return (loss, logits, labels)
        else:
            return (loss, None, None)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self.model
        model.eval()

        all_losses = []
        all_preds = []

        for step, inputs in enumerate(dataloader):
            if step >= self.eval_dataset.steps_per_epoch:
                break
            
            # print("Before prediction step: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            with torch.no_grad():
                loss, logits, _ = self.prediction_step(model, inputs, prediction_loss_only=False)
            # print("After prediction step: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

            all_losses.append(loss.item())
            if logits is not None:
                # Detach and move to CPU
                all_preds.append(logits)
                # all_preds += [p.detach().cpu() for p in logits]
                # all_preds.append(tuple(p.detach().cpu() for p in logits))
        
        # After the loop...
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            rewards_chosen = all_preds[:, 0]
            rewards_rejected = all_preds[:, 1]
            final_preds = (rewards_chosen, rewards_rejected)
            num_samples = len(rewards_chosen)
            eval_accuracy = (rewards_chosen > rewards_rejected).float().mean().item()
        else:
            final_preds = None
            num_samples = 0
            eval_accuracy = 0.0

        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
        }

        if self.model_type in ["reward", "dpo"]:
            metrics[f"{metric_key_prefix}_accuracy"] = eval_accuracy

        return EvalLoopOutput(
            predictions=final_preds,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )


    def create_model_card(
        self,
        model_name = None,
        dataset_name = None,
        tags = None,
    ):

        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Reward",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))