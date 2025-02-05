# Starbucks Representation Learning (SRL) fine-tuning for retrieval

## Installation
Our training code for passage retrieval is based on [Tevatron](https://github.com/texttron/tevatron) library.

To install Tevatron:
```bash
pip install faiss-cpu # or 'conda install pytorch::faiss-gpu' for faiss gpu search
pip install wandb # for logging
git clone https://github.com/texttron/tevatron.git
cd tevatron
pip install -e .
cd ..
```

We also use [Pyserini](https://github.com/castorini/pyserini/tree/master) to evaluate the results. 
To install it, run the following command:
```bash
conda install -c conda-forge openjdk=21 maven -y
pip install pyserini
```
If you have any issues with the pyserini installation, please follow this [link](https://github.com/castorini/pyserini/blob/master/docs/installation.md).

## Training
To train the model, run the following command:
```bash
CUDA_VISIBLE_DEVICES=1 \
python3 train.py \
  --output_dir checkpoints/retriever/bert-srl-msmarco \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --srl_training \
  --save_steps 2000 \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --pooling cls \
  --gradient_checkpointing \
  --per_device_train_batch_size 64 \
  --train_group_size 1 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --num_train_epochs 3 \
  --layer_list 2,4,6,8,10,12 \
  --embedding_dim_list 32,64,128,256,512,768 \
  --kl_divergence_weight 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --report_to wandb \
  --run_name bert-srl-msmarco
```

If you want to fine-tune with our SMAE pre-trained model, replace `bert-base-uncased` with our checkpoint here [bert-base-uncased-fineweb100bt-smae](https://huggingface.co/ielabgroup/bert-base-uncased-fineweb100bt-smae).

We also released our fine-tuned model on Hugging Face Model Hub: [Starbucks-msmarco](https://huggingface.co/ielabgroup/Starbucks-msmarco). 


## Evaluation
In this example, we use our released checkpoint [Starbucks-msmarco](https://huggingface.co/ielabgroup/Starbucks-msmarco) with dl19 dataset.
You can change `--model_name_or_path` to you own fine-tuned model.
### Step 1: Encode query and passage embeddings
#### Encode query:
```bash
CUDA_VISIBLE_DEVICES=1 \
python3 encode.py \
  --output_dir=temp \
  --model_name_or_path ./checkpoints/retriever/bert-srl-msmarco \
  --bf16 \
  --pooling cls \
  --per_device_eval_batch_size 64 \
  --query_max_len 32 \
  --passage_max_len 196 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dl19 \
  --encode_output_path embeddings/msmarco/query.dl19.pkl \
  --encode_is_query \
  --layers_to_save 2,4,6,8,10,12
```
Note, we save the full size embeddings from each target layer separately.

#### Encode passages
We shard the collection and encode each shard in parallel with multiple GPUs.
For example, if you have 2 GPUs, you can run the following commands:
```bash
mkdir -p embeddings/msmarco
NUM_AVAILABLE_GPUS=1
for i in $(seq 0 $((NUM_AVAILABLE_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${i} python encode.py \
      --output_dir=temp \
      --model_name_or_path ielabgroup/Starbucks-msmarco \
      --bf16 \
      --pooling cls \
      --per_device_eval_batch_size 64 \
      --query_max_len 32 \
      --passage_max_len 196 \
      --dataset_name Tevatron/msmarco-passage-corpus \
      --encode_output_path embeddings/msmarco/corpus.${i}.pkl \
      --layers_to_save 2,4,6,8,10,12 \
      --layer_list 2,4,6,8,10,12 \
      --embedding_dim_list 32,64,128,256,512,768 \
      --dataset_number_of_shards ${NUM_AVAILABLE_GPUS} \
      --dataset_shard_index ${i} &
    done
wait
```

### Step 2: Perform retrieval and evaluate
We perform retrieval with target layer and embedding dimensionality. 

For example, to perform retrieval with layer 6 and embedding dimension 128, run the following command:

```bash
n=6
d=128

python search.py \
--query_reps embeddings/msmarco/layer_12/query.dl19.pkl \
--passage_reps embeddings/msmarco/layer_12/"corpus*.pkl" \
--depth 1000 \
--batch_size 64 \
--save_text \
--save_ranking_to runs/run.dl19.n$n.d$d.txt \
--embedding_dim $d

# convert the results to trec format
python -m tevatron.utils.format.convert_result_to_trec \
--input runs/run.dl19.n$n.d$d.txt \
--output runs/run.dl19.n$n.d$d.trec

# Evaluation
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage runs/run.dl19.n$n.d$d.trec

Results:
ndcg_cut_10             all     0.6346
```
