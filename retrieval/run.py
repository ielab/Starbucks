import os
import subprocess
import pandas as pd

# Define parameters
layers = [2, 4, 6, 8, 10, 12]
n = 6
d = 128

results = []

for layer in layers:
    print(f"Processing layer: {layer}")
    query_reps = f"embeddings/msmarco/layer_{layer}/query.dl19.pkl"
    passage_reps = f"embeddings/msmarco/layer_{layer}/\"corpus*.pkl\""
    ranking_file = f"runs/run.dl19.n{n}.d{d}.txt"
    trec_file = f"runs/run.dl19.n{n}.d{d}.trec"

    # Run search
    search_cmd = (
        f"python search.py \
        --query_reps {query_reps} \
        --passage_reps {passage_reps} \
        --depth 1000 \
        --batch_size 64 \
        --save_text \
        --save_ranking_to {ranking_file} \
        --embedding_dim {d}"
    )
    os.system(search_cmd)

    # Convert results to TREC format
    convert_cmd = (
        f"python -m tevatron.utils.format.convert_result_to_trec \
        --input {ranking_file} \
        --output {trec_file}"
    )
    os.system(convert_cmd)

    # Evaluate results and capture output
    eval_cmd = (
        f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage {trec_file}"
    )
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
    ndcg_score = result.stdout.strip().split()[-1]  # Extract last value from output
    results.append((layer, ndcg_score))

# Create DataFrame and display results
df = pd.DataFrame(results, columns=["Layer", "NDCG Score"])
df.set_index("Layer", inplace=True)
print(df)
