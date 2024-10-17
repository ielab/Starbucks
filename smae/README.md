# Starbucks Masked Autoencoder (SMAE) Pretraining

Our SMAE is initialized with [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) 
and pre-trained on [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) datasets (100bt sample).

To do SMAE the pre-training, run the following command:
```bash
python pretrain_bert.py \
--output_dir checkpoints/bert-base-uncased-fineweb100bt-smae \
--save_steps 10000 \
--bf16 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-4 \
--lr_scheduler_type cosine \
--weight_decay 0.001 \
--warmup_ratio 0.05 \
--num_train_epochs 1 \
--logging_steps 100 \
--mlm_probability 0.2 \
--decoder_mlm_probability 0.4 \
--report_to wandb \
--matryoshka_pretraining True \
--mae_pretraining True \
--run_name bert-base-uncased-fineweb100bt-smae \
--dataloader_num_workers 16 \
--num_processes 32 \
--save_safetensors False \
--log_level info \
--logging_nan_inf_filter False
```
In our experiments, we use 8 NVIDIA H100 GPUs, resulting in a total batch size of 512 with `gradient_accumulation_steps` of 2.
If you use slurm, we also provide a script example to run the pre-training on multiple node multi gpu training: [pretrain-smae.sh](./pretrain-smae.sh).

We released our model checkpoints on Hugging Face Model Hub: [bert-base-uncased-fineweb100bt-smae](https://huggingface.co/ielabgroup/bert-base-uncased-fineweb100bt-smae). 
