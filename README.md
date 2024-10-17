# Starbucks
Starbucks: Improved Training for 2D Matryoshka Embeddings

### General guidelines
Our codebase is built on top of torch and transformers.

We recommend using a conda environment to install the required dependencies.
To install the required dependencies:

```bash
conda create -n starbucks python=3.10
conda activate starbucks

pip install torch
pip install transformers datasets peft
pip install deepspeed accelerate
```

For SMAE pre-training, see [smae](smae/README.md).

For SRL fine-tuning on retrieval task, see [retrieval](retrieval/README.md).

For SRL fine-tuning on STS task, see [sts](sts/README.md).