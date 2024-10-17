#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256g
#SBATCH --gres=gpu:2
#SBATCH --job-name=pretrain-smae
#SBATCH --output=logs/print-pretrain-smae.txt
#SBATCH --error=logs/error-pretrain-smae.txt


module load miniconda3
module load cuda
source activate starbucks # your conda env

export HF_HOME=cache/models
export HF_DATASETS_CACHE=cache/datasets
export WANDB_PROJECT=starbucks

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9909

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT pretrain_bert.py \
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
'
