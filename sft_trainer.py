import torch
import os

from transformers import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SFTTrainer(Trainer):
    """Simple SFT Trainer for supervised fine-tuning."""
    
    def __init__(self, model, args, train_dataset, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def create_model_card(
        self,
        model_name=None,
        dataset_name=None,
        tags=None,
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
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
