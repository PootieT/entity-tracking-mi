import torch

from transformers import GPT2LMHeadModel, T5ForConditionalGeneration, LlamaForCausalLM
from typing import List, Optional, Tuple, Union


class T5ForProbing(T5ForConditionalGeneration):

    # FYI: I dropped the layernorm arg here
    def __init__(self, config, probe_layer=-1):
        super().__init__(config)
        self.probe_layer = config.num_layers if probe_layer == -1 else probe_layer
        assert self.probe_layer <= config.num_layers and self.probe_layer >= 0, "Invalid layer index to probe"


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_all_layers: Optional[bool] = None
    ):

        output = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            head_mask = head_mask,
            decoder_head_mask = decoder_head_mask,
            cross_attn_head_mask = cross_attn_head_mask,
            encoder_outputs = encoder_outputs,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = True,
            return_dict = return_dict
        )
        if return_all_layers:
            return output.decoder_hidden_states
        else:
            return output.decoder_hidden_states[self.probe_layer - 1]


class T5ForIntervention(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)


class GPTForProbing(GPT2LMHeadModel):
    def __init__(self, config, probe_layer=-1):
        super().__init__(config)
        self.probe_layer = config.n_layer if probe_layer == -1 else probe_layer
        assert self.probe_layer <= config.n_layer and self.probe_layer >= 0, "Invalid layer index to probe"


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_all_layers: Optional[bool] = None
    ):

        output = super().forward(
            input_ids = input_ids,
            past_key_values = past_key_values,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = True,
            return_dict = return_dict
        )
        if return_all_layers:
            return output.hidden_states
        else:
            return output.hidden_states[self.probe_layer - 1]

class LlamaForProbing(LlamaForCausalLM):
    def __init__(self, config, probe_layer=-1):
        super().__init__(config)
        self.probe_layer = config.num_hidden_layers if probe_layer == -1 else probe_layer
        assert self.probe_layer <= config.num_hidden_layers and self.probe_layer >= 0, "Invalid layer index to probe"


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_all_layers: Optional[bool] = None
    ):

        output = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            labels = labels,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = True,
            return_dict = return_dict
        )
        if return_all_layers:
            return output.hidden_states
        else:
            return output.hidden_states[self.probe_layer - 1]