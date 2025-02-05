import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from tevatron.retriever.arguments import DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from arguments import SrlDenseModelArguments as ModelArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import EncodeCollator
from modeling import SrlDenseModel

logger = logging.getLogger(__name__)



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    layers_to_save = model_args.layers_to_save
    if layers_to_save is not None:
        layers_to_save = set(map(int, layers_to_save.split(',')))


    encode_output_folder, file_name = data_args.encode_output_path.rsplit('/', 1)

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32


    model = SrlDenseModel.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = EncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    # define encoded based on the number of layers [ [] * num_layers ]

    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    q_reps = model.encode_query(qry=batch)
                    # model output now is shape [num_layers, batch, hidden_size]
                    encoded.append(q_reps.cpu().detach().numpy())
                else:
                    p_reps = model.encode_passage(psg=batch)
                    encoded.append(p_reps.cpu().detach().numpy())

    # now encoded is a list of [num_layers, batch, hidden_size]
    # need to concatenate on dimention 1 to get [num_layers, batch, hidden_size]
    encoded = np.concatenate(encoded, axis=1)

    # for each layer, we need to pickle dump to different output_path
    for i, encoded_layer in enumerate(encoded):
        print(encoded_layer.shape)######################################################################
        out_folder = os.path.join(encode_output_folder, f"layer_{i}")
        if (layers_to_save is None) or (i in layers_to_save):
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)
            with open(os.path.join(encode_output_folder, f"layer_{i}", file_name), 'wb') as f:
                pickle.dump((encoded_layer, lookup_indices), f)


if __name__ == "__main__":
    main()

