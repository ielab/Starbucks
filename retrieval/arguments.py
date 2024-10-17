import logging
from tevatron.retriever.arguments import ModelArguments
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class SrlDenseModelArguments(ModelArguments):
    kl_divergence_weight: float = field(default=0.0, metadata={"help": "KL divergence weight for russian doll training"})
    layer_list: Optional[str] = field(default=None, metadata={"help": "Comma-separated list of layer indices to save. Example: '2,6,12'"})
    embedding_dim_list: Optional[str] = field(default=None, metadata={"help": "Comma-separated list of layer indices to save. Example: '32,128,256,768'"})
    sub_model_sampling: bool = field(default=False, metadata={"help": "default True: sample one layer from layer_list, "
                                                                     "compute losses for all sub-embeddings of the full layer and sampled layer."})
    srl_training: bool = field(default=False, metadata={"help": "Russian doll training"})
    layers_to_save: Optional[str] = field(default=None, metadata={
        "help": "Comma-separated list of layer indices to save. Example: '0,2,4'"})
