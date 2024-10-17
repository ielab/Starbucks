"""
The file trains for multiple baseline models using all_nli or low_resource datasets (only stsb) and saves the model in the output directory.
Usage:
python train_baseline.py

OR
python train_starbucks.py pretrained_transformer_model_name training_type

"""
import logging
import os
import sys
import traceback
from datetime import datetime
from tqdm import tqdm

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction


model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
training_type = sys.argv[2] if len(sys.argv) > 2 else "full"



dims = []
layers = []


if training_type == "full":
    batch_size = 128
    gradient_accumulation_steps = 1
    num_train_epochs = 1
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    output_parent= f"all_nli/{model_name.replace('/', '-')}/baselines"

elif training_type == "low":
    batch_size = 16
    gradient_accumulation_steps = 1
    num_train_epochs = 4
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    output_parent = f"low_resource/{model_name.replace('/', '-')}/baselines"

# for i in range(2, 13, 2):
#     for j in [64, 32]:
#         if os.path.exists(f"{output_parent}/layer_{i}_dim_{j}/final"):
#             continue
#         layers.append(i)
#         dims.append(j)

dims += [32, 64, 128, 256, 512, 768]
layers += list(range(2, 13, 2))

test_dataset = load_dataset("sentence-transformers/stsb", split="test")


for layer_idx, layer in enumerate(layers):
    # for dim in dims:
    # Save path of the model
    dim = dims[layer_idx]
    output_dir = f"{output_parent}/layer_{layer}_dim_{dim}"

    model = SentenceTransformer(model_name)
    model._first_module().auto_model.encoder.layer = model._first_module().auto_model.encoder.layer[:layer]

    logging.info(model)

    logging.info(train_dataset)

    if training_type == "full":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif training_type == "low":
        train_loss = losses.CoSENTLoss(model)


    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_steps=1000,
        run_name=f"sts-{layer}-{dim}",  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,

    )
    trainer.train()

    # 8. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final"
    model.save(final_output_dir)
