import os
import pdb
import glob
import argparse
import collections
import json

from tqdm import tqdm
import pandas as pd



def compute_intervention_metrics(output_path):
    correct = 0
    total = 0
    with open(output_path) as f:
        for line in f:
            d = json.loads(line)
            model_answer = d["intervened_answer"].split(", Box")[0].strip().rstrip(".")
            if model_answer.startswith("</s>"):
                model_items = []
            else:
                model_items = model_answer.lower().split(" and ")
                model_items = [i.removeprefix("the ") for i in model_items]
            print("Model answer: ", model_answer)
            print("Model items: ", model_items)
            target_items = d["orig_items"]
            print("Target items: ", target_items)
            print()
            if collections.Counter(model_items) == collections.Counter(target_items):
                correct += 1
            total += 1
    print(correct/total, correct, total)

def get_prompt_name_from_file_name(file_name:str):
    start_idx = file_name.find("_fs")+3
    idx = start_idx
    while idx < len(file_name) and (file_name[idx].isupper() or file_name[idx]=="_"):
        idx += 1
    return file_name[start_idx:idx-1]

def compute_inference_metrics_by_ops(output_path, log: bool=True):
    if os.path.exists(output_path):
        df_paths = {output_path: pd.read_json(output_path, lines=True)}
    else:
        df_paths = {p: pd.read_json(p, lines=True) for p in glob.glob(output_path)}

    agg_results = []
    for path, df in tqdm(df_paths.items()):
        if log:
            print(f"Computing inference metrics for {path} ===============")
        # first non-aggregated by datapoints (7 sentence per datapoint)
        # print(df.groupby(["numops"])["correct"].mean())
        acc = df.groupby(["numops"]).agg(accuracy=pd.NamedAgg("correct", lambda x: x.mean()),
                                         count=pd.NamedAgg("correct", lambda x: x.count()),
                                         correct=pd.NamedAgg("correct", lambda x: x.sum())).transpose()
        if log:
            print(acc)

        # then aggregate by datapoints
        df["context"] = df.prefix.apply(lambda x: " ".join(x.split(" ")[:-3]))
        context_accuracy = (
            df.groupby("context")["correct"]
            .apply(lambda x: 1 if x.all() else 0)
            .reset_index(name="context_accuracy")
        )
        context_numops = (
            df.groupby("context")["numops_global"]
            .apply(lambda x: list(x)[0])
            .reset_index(name="context_numops")
        )
        df_join = pd.merge(context_accuracy, context_numops, on="context")
        agg_acc = df_join.groupby(["context_numops"]).agg(accuracy=pd.NamedAgg("context_accuracy", lambda x: x.mean()),
                                         count=pd.NamedAgg("context_accuracy", lambda x: x.count()),
                                         correct=pd.NamedAgg("context_accuracy", lambda x: x.sum())).transpose()
        if log:
            print(agg_acc)
            print()

        # aggregate results
        file_name = path.split("/")[-1]
        model_name = file_name.split("_")[1]
        bits = 8 if "8bit" in file_name else 4 if "4bit" in file_name else 16
        prompt = get_prompt_name_from_file_name(file_name) if "_fs" in file_name else ""
        chat = True if "_chat_" in file_name else False
        row = {
            "model_name": model_name,
            "bits": bits,
            "prompt": prompt,
            "chat": chat,
        }
        row.update(acc.loc['accuracy'].to_dict())
        agg_results.append(row)

    agg_results = pd.DataFrame(agg_results).sort_values(["model_name", "prompt", "chat"])
    print(agg_results)
    print(acc)
    agg_results.to_csv("inference_results_aggregated.csv", index=False)
    return agg_results

def main():
    """
        Command line utility.
    """

    parser = argparse.ArgumentParser("CLI utility for evaluating model outputs.")
    parser.add_argument("-v", "--verbose", action="store_true",)
    parser.add_argument(
        "-o","--model_output", type=str,
        default="/projectnb/mcnet/peter/entity-tracking-probing/results/boxes_altAlways_default_maxop12_5k/baseline_inference/*.jsonl",
        help="Path or pattern to model output in jsonl format.",
    )
    args = parser.parse_args()
    compute_inference_metrics_by_ops(args.model_output, args.verbose)


if __name__ == "__main__":
    main()

