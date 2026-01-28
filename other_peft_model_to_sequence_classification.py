from peft import (
    PeftModelForSequenceClassification,
    PeftConfig
)
from transformers import (
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch



class ConvertedPeftModelForSequenceClassification(PeftModelForSequenceClassification):
    def __init__(self, peft_config: PeftConfig, model: AutoModelForCausalLM, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.num_labels = model.config.num_labels
        self.problem_type = "regression"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs):
        
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        # Extract logits from the outputs
        logits = outputs.logits

        # select last "real" token and ignore padding tokens

        sequence_lengths   = torch.sum(attention_mask, dim=1)
        last_token_indices = sequence_lengths - 1
        batch_size         = logits.shape[0]
       
        # Get the logits for the last token in the sequence
        logits = logits[torch.arange(batch_size, device=logits.device), last_token_indices, :]
        #logits = logits[:, -1, :] # if batch_size = 1

        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)