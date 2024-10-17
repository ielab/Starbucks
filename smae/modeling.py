from transformers import Trainer, PreTrainedTokenizer, BertPreTrainedModel, DataCollatorForWholeWordMask, AutoTokenizer, BertForMaskedLM, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import MaskedLMOutput, BertModel, BertOnlyMLMHead

from typing import List, Tuple, Dict, Any, Optional, Union
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
import random
from torch.nn import functional as F
import collections
import copy
from torch import nn


class BertPredictionHeadTransformForwardDecorator:
    def __init__(self, module):
        self.module = module

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dim = hidden_states.size(-1)
        hidden_states = F.linear(hidden_states,
                                 self.module.dense.weight[:, :dim],
                                 self.module.dense.bias)
        hidden_states = self.module.transform_act_fn(hidden_states)

        hidden_states = self.module.LayerNorm(hidden_states)

        return hidden_states


class BertFor2DMatryoshkaMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.cls.predictions.transform.forward = BertPredictionHeadTransformForwardDecorator(
            self.cls.predictions.transform)

        self.layer_list = [2, 4, 6, 8, 10, 12]
        self.dim_list = [32, 64, 128, 256, 512, 768]


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states

        total_loss = 0
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            for selected_layer, selected_dim in zip(self.layer_list, self.dim_list):
                selected_layer_selected_dim_scores = self.cls(hidden_states[selected_layer][:, :, :selected_dim])
                selected_layer_selected_dim_loss = loss_fct(selected_layer_selected_dim_scores.view(-1, self.config.vocab_size),
                                                            labels.view(-1))
                total_loss += selected_layer_selected_dim_loss

            total_loss /= len(self.layer_list)


        if not return_dict:
            output = (None,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MaskedLMOutput(
            loss=total_loss,
            logits=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertFor2DMaekMatryoshkaMaskedLM(BertFor2DMatryoshkaMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        n_head_layers = 1 # hard code for now
        self.decoder = BertForMaskedLM.from_pretrained(config._name_or_path)
        self.decoder.cls = self.cls
        self.decoder.bert.embeddings = self.bert.embeddings
        self.decoder.bert.encoder.layer = self.decoder.bert.encoder.layer[:n_head_layers]


    def compute_mae_loss(self, hidden_states, decoder_input_ids, decoder_labels, decoder_attention_mask, loss_fct):
        encoder_cls_hiddens = hidden_states[:, :1]
        decoder_input_embeds = self.decoder.bert.embeddings(decoder_input_ids)
        decoder_input_embeds[:, :1] = encoder_cls_hiddens
        decoder_input_mlm = self.decoder.bert.encoder(decoder_input_embeds, attention_mask=decoder_attention_mask)[0]

        mae_scores = self.decoder.cls(decoder_input_mlm)
        mae_loss = loss_fct(mae_scores.view(-1, self.config.vocab_size),
                            decoder_labels.view(-1))
        return mae_loss


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states
        decoder_attention_mask = self.decoder.get_extended_attention_mask(
            attention_mask,
            attention_mask.shape,
            attention_mask.device
        )

        total_loss = 0
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            for selected_layer, selected_dim in zip(self.layer_list, self.dim_list):
                selected_layer_selected_dim_scores = self.cls(hidden_states[selected_layer][:, :, :selected_dim])
                selected_layer_selected_dim_loss = loss_fct(selected_layer_selected_dim_scores.view(-1, self.config.vocab_size),
                                                            labels.view(-1))

                mae_loss = self.compute_mae_loss(self.cls.predictions.transform(hidden_states[selected_layer][:, :, :selected_dim]),
                                                 decoder_input_ids,
                                                 decoder_labels,
                                                 decoder_attention_mask,
                                                 loss_fct)
                total_loss += (selected_layer_selected_dim_loss + mae_loss) / 2

            total_loss /= len(self.layer_list)





        if not return_dict:
            output = (None,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MaskedLMOutput(
            loss=total_loss,
            logits=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )