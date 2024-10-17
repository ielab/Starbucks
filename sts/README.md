# Starbucks SRL fine-tuning for STS

This repository contains the code for fine-tuning from any pre-trained model on the STS benchmark dataset.
This repo supports for three types of fine-tuning:
- **baseline**: fine-tuning the small scale models seperatly
- **2d_matryoshka**: fine-tuning the the full scale model with 2d matryoshka
- **starbucks**: fine-tuning the full scale model with starbucks representation learning

## To train models

You can train the models by running the following commands:
```bash
python3 train_baseline.py bert-base-uncased full # full means use all_nli, otherwise only stab to train

python3 train_2d_matryoshka.py bert-base-uncased full # full means use all_nli, otherwise only stab to train

python3 train_starbucks.py bert-base-uncased full 1 # full means use all_nli, otherwise only stab to train, 1 means kl_divergence weight
```
You can change the model name to any other pre-trained model name in the huggingface model hub, or local path to the model.

## To evaluate models
```bash
python3 inference_2d_sts.py [model_name] full diaganol # full means use all_nli, otherwise only stab to train, diaganol means only starbucks sizes
```

Or to evaluate all seperatly trained models at the same time:
```bash
python3 inference_baselines_sts.py [model_name] full # full means use all_nli, otherwise only stab to train
```

We released our model checkpoints on Hugging Face Model Hub: [Starbucks_STS](https://huggingface.co/ielabgroup/Starbucks_STS). 



