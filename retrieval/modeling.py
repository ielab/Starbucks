import torch
import logging
from tevatron.retriever.modeling.encoder import EncoderOutput
from tevatron.retriever.modeling.dense import DenseModel
from typing import Dict, Optional
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from arguments import SrlDenseModelArguments
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from typing import Dict, List
from torch import nn, Tensor

import random
import torch.distributed as dist
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class SrlDenseModel(DenseModel):

    def __init__(self,
                 kl_divergence_weight: float = 0.0,
                 layer_list: List = None,
                 embedding_dim_list: List = None,
                 sub_model_sampling: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.kl_divergence_weight = kl_divergence_weight
        self.sub_model_sampling = sub_model_sampling

        if layer_list is not None:
            self.layer_list = [int(i) for i in layer_list]
        else:
            self.layer_list = [-1]

        if embedding_dim_list is not None:
            self.embedding_dim_list = [int(i) for i in embedding_dim_list]
        else:
            self.embedding_dim_list = [-1]


    def gradient_checkpointing_enable(self, **kwargs):
        # check if the encoder has gradient_checkpointing_enable method
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.model.gradient_checkpointing_enable()

    def encode_query(self, qry):
        query_output = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        all_hidden_states = torch.stack(query_output.hidden_states)  # Note the layer 0 is embedding layer
        return self._pooling(all_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg):
        # Encode passage is the same as encode query
        return self.encode_query(psg)

    def _pooling(self, all_hidden_states, attention_mask):
        # all_hidden_states shape: [num_layers, batch_size, seq_length, hidden_dim]
        if self.pooling in ['cls', 'first']:
            # Select the first token for all layers
            reps = all_hidden_states[:, :, 0, :]  # Shape: [num_layers, batch_size, hidden_dim]
        elif self.pooling in ['mean', 'avg', 'average']:
            expanded_mask = attention_mask[None, ... , None]
            masked_hiddens = all_hidden_states.masked_fill(~expanded_mask.bool(), 0.0)
            reps = masked_hiddens.sum(dim=2) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            reps = all_hidden_states[:, torch.arange(all_hidden_states.size(1)), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        # reps dimensions: [num_layers, batch_size, hidden_dim]
        return reps

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=1)

        return all_tensors


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        total_loss = 0 if self.training else None

        if self.training:
            q_reps = q_reps[self.layer_list]
            p_reps = p_reps[self.layer_list]

            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            # Assume q_reps and p_reps are of shape [num_layers, batch_size, hidden_size]
            num_layers, batch_size, embedding_dim = q_reps.shape
            target = torch.arange(q_reps.size(1), device=q_reps.device, dtype=torch.long)
            target = target * (p_reps.size(1) // q_reps.size(1))

            # first is to compute the element-wise scores for each layer and full dim
            all_scores = torch.einsum('lqh,ldh -> lqdh', q_reps, p_reps) # [num_layers, batch_size (num_queries), num_docs, hidden_size]

            # full layer full dim predicted scores and loss
            last_layer_index = num_layers - 1
            full_layer_full_dim_scores = all_scores[last_layer_index, :, :]
            full_layer_scores = full_layer_full_dim_scores.sum(dim=-1)  # [batch_size, num_docs], dot product scores
            total_loss = self.compute_loss(full_layer_scores / self.temperature, target)

            target_scores = None
            if self.kl_divergence_weight > 0:
                target_scores = full_layer_scores.detach()

            if self.sub_model_sampling:
                layer_indices = random.sample(range(num_layers-1), 1) # sample one layer except the full layer
                layer_indices.append(last_layer_index) # add the full layer index
                layer_dim_tuples = [(layer_idx, dim) for layer_idx in layer_indices for dim in self.embedding_dim_list]
            else:
                assert num_layers == len(self.embedding_dim_list), 'embedding_dim_list should have the same length as num_layers'
                layer_dim_tuples = [(layer_idx, dim) for layer_idx, dim in zip(range(num_layers), self.embedding_dim_list)]

            for layer_idx, dim in layer_dim_tuples:
                if layer_idx == last_layer_index and dim == embedding_dim:
                    # full layer full dim loss is already computed
                    continue
                layer_dim_scores = all_scores[layer_idx, :, :, :dim]  # [batch_size, num_docs, dim]
                layer_scores = layer_dim_scores.sum(dim=-1)  # dot product
                total_loss += self.compute_loss(layer_scores / self.temperature, target)
                if self.kl_divergence_weight > 0:
                    kl_loss = F.kl_div(F.log_softmax(layer_scores / self.temperature, dim=1),
                                       F.softmax(target_scores / self.temperature, dim=1),
                                       reduction='batchmean')
                    total_loss += self.kl_divergence_weight * kl_loss

        else:
            raise NotImplementedError('Evaluation mode not implemented yet')

        return EncoderOutput(
            loss=total_loss,
            scores=full_layer_scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @classmethod
    def build(
            cls,
            model_args: SrlDenseModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_divergence_weight=model_args.kl_divergence_weight,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None,
                sub_model_sampling=model_args.sub_model_sampling
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                kl_divergence_weight=model_args.kl_divergence_weight,
                layer_list=model_args.layer_list.split(',') if model_args.layer_list else None,
                embedding_dim_list=model_args.embedding_dim_list.split(',') if model_args.embedding_dim_list else None,
                sub_model_sampling=model_args.sub_model_sampling
            )
        return model

