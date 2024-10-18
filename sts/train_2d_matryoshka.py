"""
The file trains for 2dmse model using all_nli or low_resource datasets (only stsb) and saves the model in the output directory.
Usage:
python 2d_matryoshka.py

OR
python 2d_matryoshka.py pretrained_transformer_model_name training_data
"""

import logging
import sys

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased" # model name
training_data = sys.argv[2] if len(sys.argv) > 2 else "full" # full or low, full for all-nli and low for stsb



# 1. load the dataset
if training_data == "full":
    batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
    gradient_accumulation_steps = 1
    num_train_epochs = 1
    train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    output_dir = f"all_nli/{model_name.replace('/', '-')}/2d"
else:
    batch_size = 16  # The larger you select this, the better the results (usually). But it requires more GPU memory
    gradient_accumulation_steps = 1
    num_train_epochs = 4
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    output_dir = f"low_resource/{model_name.replace('/', '-')}/2d"


# 2. Load the model
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75

logging.info(model)
logging.info(train_dataset)

# 3. Define our training loss
if training_data == "full":
    inner_train_loss = losses.MultipleNegativesRankingLoss(model)
else:
    inner_train_loss = losses.CoSENTLoss(model)

train_loss = losses.Matryoshka2dLoss(model, inner_train_loss, [768, 512, 256, 128, 64, 32])

# 4. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    warmup_ratio=0.1,
    seed=42,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    save_strategy="steps",
    save_total_limit=2,
    run_name="2d-matryoshka-nli",  # Will be used in W&B if `wandb` is installed
)

# 5. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)
trainer.train()

# 6. Evaluate the model performance on the STS Benchmark test dataset
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 7. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)