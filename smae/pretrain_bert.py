import logging
import os
import sys

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from transformers import Trainer, AutoTokenizer, BertForMaskedLM
import torch
from dataclasses import dataclass, field
from modelling import BertFor2DMatryoshkaMaskedLM, BertFor2DMaekMatryoshkaMaskedLM
from data import MLMDataset, DataCollatorForWholeWordMaskWithAttentionMask, MaeDataCollatorForWholeWordMask

logger = logging.getLogger(__name__)


class SkipNanTrainer(Trainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        # Immediately check for NaN gradients
        nan_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_gradients = True
                param.grad = None  # Reset gradients, so that the optimizer.step() later will do nothing.

        if nan_gradients:
            # Tip: set '--logging_nan_inf_filter False' for smooth logging
            print("NaN gradient detected, skipping optimizer step.")

        return loss
    

@dataclass
class MatryoshkaPretrainingArguments(TrainingArguments):
    matryoshka_pretraining: bool = field(default=False, metadata={"help": "Do matryoshka pretraining"})
    mae_pretraining: bool = field(default=False, metadata={"help": "Do MAE pretraining"})
    num_processes: int = field(default=16, metadata={"help": "Number of processes to use for data loading"})
    mlm_probability: float = field(default=0.15, metadata={"help": "Probability of masking tokens"})
    decoder_mlm_probability: float = field(default=0.3, metadata={"help": "Probability of masking tokens for the decoder"})



def main():
    parser = HfArgumentParser((MatryoshkaPretrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, = parser.parse_args_into_dataclasses()
        training_args: MatryoshkaPretrainingArguments
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = MLMDataset(tokenizer, training_args.num_processes)

    if training_args.matryoshka_pretraining:
        if training_args.mae_pretraining:
            collator = MaeDataCollatorForWholeWordMask(tokenizer,
                                                       encoder_mlm_probability=training_args.mlm_probability,
                                                       decoder_mlm_probability=training_args.decoder_mlm_probability,
                                                       pad_to_multiple_of=8)
            model = BertFor2DMaekMatryoshkaMaskedLM.from_pretrained('bert-base-uncased')
        else:
            collator = DataCollatorForWholeWordMaskWithAttentionMask(tokenizer,
                                                                     mlm_probability=training_args.mlm_probability,
                                                                     pad_to_multiple_of=8)
            model = BertFor2DMatryoshkaMaskedLM.from_pretrained('bert-base-uncased')
    else:
        collator = DataCollatorForWholeWordMaskWithAttentionMask(tokenizer,
                                                                 mlm_probability=training_args.mlm_probability,
                                                                 pad_to_multiple_of=8)
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')


    trainer = SkipNanTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    train_dataset.trainer = trainer

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
