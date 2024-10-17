# Starbucks Masked Autoencoder

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
--run_name pretrain-bert-mae-matryoshka-fineweb100bt-starbucks \
--dataloader_num_workers 16 \
--num_processes 32 \
--save_safetensors False \
--log_level info \
--logging_nan_inf_filter False
```
