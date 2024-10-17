import json
import os.path
import sys
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
import re

model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased" # model name, could be any from huggingface
evaluate_type = sys.argv[2] if len(sys.argv) > 2 else "full" # full or low
layer_dim_type = sys.argv[3] if len(sys.argv) > 3 else "diaganol" # diaganol or full
if evaluate_type == "full":
    dataset_dict = {
        "stsb": "sentence-transformers/stsb",
        "sts12": "mteb/sts12-sts",
        "sts13": "mteb/sts13-sts",
        "sts14": "mteb/sts14-sts",
        "sts15": "mteb/sts15-sts",
        "sts16": "mteb/sts16-sts",
        "sickr": "mteb/sickr-sts"
    }
elif evaluate_type == "low":
    dataset_dict = {
        "stsb": "sentence-transformers/stsb",
    }

final_result_dict = {}

matryoshka_dims = []

matryoshka_layers = []

matryoshka_dims += [768, 512, 256, 128, 64, 32]
matryoshka_layers += [12, 10, 8, 6, 4, 2]

for dataset in tqdm(dataset_dict.keys()):
    dataset_loading_name = dataset_dict[dataset]
    test_dataset = load_dataset(dataset_loading_name, split="test")
    result_dict = {}
    for layer_i, layer in enumerate(matryoshka_layers):
        evaluators = []
        for dim in matryoshka_dims:
            if layer_dim_type == "diaganol":
                if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
                    continue
            evaluators.append(
                EmbeddingSimilarityEvaluator(
                    sentences1=test_dataset["sentence1"],
                    sentences2=test_dataset["sentence2"],
                    scores=test_dataset["score"],
                    main_similarity=SimilarityFunction.COSINE,
                    name=f"sts-test-{dim}",
                    truncate_dim=dim
                )
            )
        model = SentenceTransformer(model_name)
        model[0].auto_model.encoder.layer = model[0].auto_model.encoder.layer[:layer]
        test_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
        results = test_evaluator(model)

        result_dict[layer] = {}
        for result_key in list(results.keys()):
            if "spearman_cosine" in result_key:
                # first copy the key
                #result_key save is only the number in the key
                result_key_save = re.findall(r'\d+', result_key)[0]
                #print(result_key_save)
                result_dict[layer][result_key_save] = results[result_key]
    final_result_dict[dataset] = result_dict

final_result_dict["average"] = {}
for layer_i, layer in enumerate(matryoshka_layers):
    final_result_dict["average"][layer] = {}
    #dim = matryoshka_dims[layer_i]
    for dim in matryoshka_dims:
        if layer_dim_type == "diaganol":
            if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
                continue
        final_result_dict["average"][layer][dim] = sum([final_result_dict[dataset][layer][str(dim)] for dataset in dataset_dict.keys()]) / len(dataset_dict.keys())

final_result_dict["average_dataset"] = {}
for dataset in dataset_dict.keys():
    final_result_dict["average_dataset"][dataset] = []
    for layer_i, layer in enumerate(matryoshka_layers):
        for dim in matryoshka_dims:
            if layer_dim_type == "diaganol":
                if matryoshka_dims.index(dim) != matryoshka_layers.index(layer):
                    continue
            final_result_dict["average_dataset"][dataset].append(final_result_dict[dataset][layer][str(dim)])
    final_result_dict["average_dataset"][dataset] = sum(final_result_dict["average_dataset"][dataset]) / len(final_result_dict["average_dataset"][dataset])

model_output_folder = model_name.replace("/", "_")
if not os.path.exists(model_output_folder):
    os.makedirs(model_output_folder)

out_file = os.path.join(model_output_folder, "sts_results_" + evaluate_type + "_" + layer_dim_type + ".json")
json.dump(final_result_dict, open(out_file, "w"), indent=2)




