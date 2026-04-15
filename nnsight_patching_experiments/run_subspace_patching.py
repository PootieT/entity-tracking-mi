import json
import os
from collections import defaultdict
from functools import partial
from itertools import product, chain
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Literal
import argparse

from tqdm import tqdm

from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import nnsight
from nnsight import LanguageModel
# these libraries has to be imported after nnsight
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..")
from utils import fix_random_seed, free_gpu_cache, get_basis_directions, pad_batch_collate_fn, setup_nnsight
from patch_utils import build_parser, post_arg_parse_fix, get_model_and_dataset



hypothesis_to_intervention_positions = {
    # since last_token is different actual index across data points, use physical string and resolve later
    # currently resolution strategy is corresponding to a field in the batch. But later may need to dynamically
    # compute it.
    "obj": {"cache": ["last_token"], "patch": ["last_token"]},
    "pos_phrase_ctf_op": {"cache": ["last_token"], "patch": ["last_token"]}
}

def fix_fonts(title=20, label=20, xtick=15, ytick=15, default=15):
    # Set the global font family to 'Times New Roman'
    # keep running into
    plt.rc('font', family='serif', serif=['Times New Roman'])

    # Set the global default font size (e.g., to 14)
    plt.rcParams["font.size"] = default
    plt.rcParams["xtick.labelsize"] = xtick  # Optional: specific size for x-axis ticks
    plt.rcParams["ytick.labelsize"] = ytick  # Optional: specific size for y-axis ticks
    plt.rcParams["axes.labelsize"] = label  # Optional: specific size for axis labels
    plt.rcParams["axes.titlesize"] = title  # Optional: specific size for plot titles



def split_train_validation_data(
    dataset: Dataset, tokenizer, args: argparse.Namespace
) -> Tuple[DataLoader, DataLoader]:
    assert len(dataset) >= (args.train_size + args.validation_size)
    train_dataset = dataset.select(range(args.train_size))
    val_dataset = dataset.select(range(args.train_size, args.train_size+args.validation_size))
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=partial(pad_batch_collate_fn, tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=partial(pad_batch_collate_fn, tokenizer=tokenizer))
    return train_loader, val_loader


def validate(
    exp_name: Literal["obj", "pos_phrase_ctf_op"],
    lm: LanguageModel,
    layer_idx: int,
    validation_loader: DataLoader,
    projection: torch.Tensor | dict[str, torch.Tensor] | None = None,
    verbose: bool = False,
    remote: bool = False,
) -> Tuple[float, List[List[int]], List[List[int]]]:

    intervention_positions = hypothesis_to_intervention_positions[exp_name]
    patch_to_cache_map = {k: v for k, v in zip(intervention_positions["patch"], intervention_positions["cache"])}
    total = 0
    argmax_correct_any = 0
    argmax_correct_full = []
    topk_correct_full = []
    for batch_idx, batch in tqdm(enumerate(validation_loader), total=len(validation_loader)):
        alt_prompts = batch["source_tokens"]  # TODO misnomer, actually tokens not prompts
        org_prompts = batch["base_tokens"]
        target_tokens = batch["source_labels"]
        max_n_labels = max(len(l) for l in target_tokens)
        batch_last_token_indices = batch[f"base_last_token_indices"]

        batch_size = len(target_tokens)
        alt_acts = defaultdict(dict)

        def nnsight_request():
            with torch.no_grad():
                with lm.trace(remote=remote) as tracer:
                    with tracer.invoke(alt_prompts):
                        for alt_token_position in intervention_positions["cache"]:
                            batch_alt_token_indices = batch[f"source_{alt_token_position}_indices"]
                            # alt_acts[alt_token_position] = lm.model.layers[layer_idx].output[0][:, batch_alt_token_indices].clone()
                            alt_acts[alt_token_position] = lm.model.layers[layer_idx].output[:, batch_alt_token_indices].clone()

                    with tracer.invoke(org_prompts):
                        for org_token_position in intervention_positions["patch"]:
                            batch_org_token_indices = batch[f"base_{org_token_position}_indices"]
                            # curr_output = lm.model.layers[layer_idx].output[0][:, batch_org_token_indices].clone()
                            curr_output = lm.model.layers[layer_idx].output[:, batch_org_token_indices].clone()
                            if projection is not None:
                                proj = projection
                                alt_proj = torch.matmul(alt_acts[patch_to_cache_map[org_token_position]], proj)
                                org_proj = torch.matmul(curr_output, proj)
                                patch = curr_output - org_proj + alt_proj

                                del alt_proj, org_proj
                                free_gpu_cache()
                            else:
                                patch = alt_acts[patch_to_cache_map[org_token_position]]

                            # lm.model.layers[layer_idx].output[0][:, batch_org_token_indices] = patch
                            lm.model.layers[layer_idx].output[:, batch_org_token_indices] = patch

                        logits = lm.lm_head.output
                        last_token_logits = logits[range(batch_size), batch_last_token_indices]
                        topk_pred = last_token_logits.argsort(dim=-1, descending=True)[:, :max_n_labels].cpu().numpy().save()

                return topk_pred

        topk_pred = nnsight_request()

        for i in range(batch_size):

            labels = target_tokens[i]  # multiple target objects
            label_texts = [lm.tokenizer.decode(l).strip().lower() for l in labels]
            topk_pred_texts = [lm.tokenizer.decode(l).strip().lower() for l in topk_pred[i, :len(label_texts)]]

            if topk_pred_texts[0] in label_texts:
                argmax_correct_any += 1

            argmax_correct_full_batch = []
            topk_correct_full_batch = []
            for k, label_text in enumerate(label_texts):
                argmax_correct_full_batch.append(1 if label_text == topk_pred_texts[0] else 0)
                topk_correct_full_batch.append(1 if label_text in topk_pred_texts else 0)

            argmax_correct_full.append(argmax_correct_full_batch)
            topk_correct_full.append(topk_correct_full_batch)
            total += 1
            # if verbose:
                # print(f"Correct: {topk_pred_texts[0] in label_texts} | Predicted: {topk_pred_texts} | Target: {label_texts}")

        del alt_acts, alt_prompts, org_prompts, target_tokens, topk_pred
        free_gpu_cache()

    argmax_correct_any = argmax_correct_any / total
    return argmax_correct_any, argmax_correct_full, topk_correct_full


def get_low_rank_projection(
    exp_name: Literal["obj", "pos_phrase_ctf_op", "pos_phrase_og_op", "pos_phrase_legal_op"],
    lm: LanguageModel,
    layer_idx: int,
    train_loader: DataLoader,
    basis_directions: Float[Tensor, "n_basis n_basis"],
    learning_rate: float = 0.1,
    n_epochs: int = 1,
    lamb: float = 0.1,
    target_object_types: Optional[List[str]] = None,
    remove_op_order: bool = True,
    verbose: bool = False,
    remote: bool = False,
) -> tuple[torch.Tensor, dict]:
    if remote:
        raise NotImplementedError("Training not tested for remote yet")

    intervention_positions = hypothesis_to_intervention_positions[exp_name]
    patch_to_cache_map = {k: v for k, v in zip(intervention_positions["patch"], intervention_positions["cache"])}

    basis_indices = list(range(basis_directions[layer_idx].size(0)))
    mask = torch.ones(len(basis_indices), requires_grad=True, device="cuda", dtype=torch.bfloat16)
    basis_directions = basis_directions.to("cuda")
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    # training loop
    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            alt_prompts = batch["source_tokens"] # TODO misnomer, actually tokens not prompts
            org_prompts = batch["base_tokens"]
            target_tokens = batch["source_labels"]
            batch_last_token_indices = batch[f"base_last_token_indices"]
            if remove_op_order:
                # remove operation index information in label types (i.e. "op1_put" -> "put")
                batch[f"source_label_types"] = [[l.split("_")[-1] for l in sample_label_types] for sample_label_types in batch[f"source_label_types"]]

            batch_size = len(target_tokens)
            alt_acts = defaultdict(dict)

            if target_object_types is not None:
                filtered_target_tokens = []
                for i in range(batch_size):
                    sample_object_types = batch[f"source_label_types"][i]
                    sample_target_tokens = [tok for tok_idx, tok in enumerate(target_tokens[i]) if sample_object_types[tok_idx] in target_object_types]
                    filtered_target_tokens.append(np.array(sample_target_tokens))
                target_tokens = filtered_target_tokens

            masked_directions = basis_directions * mask.unsqueeze(-1)
            proj_matrix = torch.matmul(masked_directions.T, masked_directions).to(lm.dtype)

            with lm.trace(remote=remote) as tracer:
                with tracer.invoke(alt_prompts):
                    for alt_token_position in intervention_positions["cache"]:
                        batch_alt_token_indices = batch[f"source_{alt_token_position}_indices"]
                        # alt_acts[alt_token_position] = lm.model.layers[layer_idx].output[0][:, batch_alt_token_indices].clone()
                        alt_acts[alt_token_position] = lm.model.layers[layer_idx].output[:, batch_alt_token_indices].clone()

                with tracer.invoke(org_prompts):
                    for org_token_position in intervention_positions["patch"]:
                        batch_org_token_indices = batch[f"base_{org_token_position}_indices"]
                        # curr_output = lm.model.layers[layer_idx].output[0][:, batch_org_token_indices].clone()
                        curr_output = lm.model.layers[layer_idx].output[:, batch_org_token_indices].clone()
                        proj = proj_matrix
                        alt_proj = torch.matmul(alt_acts[patch_to_cache_map[org_token_position]], proj)
                        org_proj = torch.matmul(curr_output, proj)
                        # lm.model.layers[layer_idx].output[0][:, batch_org_token_indices] = (curr_output - org_proj + alt_proj)
                        lm.model.layers[layer_idx].output[:, batch_org_token_indices] = (curr_output - org_proj + alt_proj)

                    logits = lm.lm_head.output[torch.arange(batch_size), batch_last_token_indices].save()

                    del alt_acts, org_proj
                    free_gpu_cache()

            task_loss = 0
            for i in range(batch_size):
                task_loss += -torch.mean(logits[i, torch.LongTensor(target_tokens[i].tolist()).to(lm.device)])

            # target_logit = logits[torch.arange(batch_size), target_tokens]
            # task_loss = -torch.mean(target_logit)
            l1_loss = lamb * torch.norm(mask, p=1)
            loss = task_loss + l1_loss.to(task_loss.device)

            if verbose:
                mask_data = torch.round(mask.data.clone().clamp_(0, 1))
                cur_rank = mask_data.sum().item()
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Rank: {cur_rank}, Loss: {loss.item()} | l_task: {task_loss.item()}, l1: {l1_loss.item()}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Clamp after optimizer step
            with torch.no_grad():
                mask.data.clamp_(0, 1)

            epoch_loss += loss.item()

            del logits, proj_matrix
            free_gpu_cache()

    # build the projection after training
    mask_data = mask.data.clone()
    mask_data.clamp_(0, 1)
    rounded = torch.round(mask_data)

    masked_directions = basis_directions * rounded.unsqueeze(-1)
    proj_matrix = torch.matmul(masked_directions.T, masked_directions).to(lm.dtype)

    metadata = {"mask": rounded.tolist(), "rank": rounded.sum().item()}

    return proj_matrix, metadata



def load_single_experiment(
    exp_dir: str,
    metric: str="argmax_any",
    label_types: Optional[List[List[str]]]=None,
    remove_op_order:bool=False,
    load_mask:bool=False,
)-> pd.DataFrame:
    rows = []
    for layer_result_file in os.listdir(exp_dir):
        layer_idx = int(layer_result_file.split(".")[0])
        data = json.load(open(os.path.join(exp_dir, layer_result_file)))
        row = {
            "layer": layer_idx,
            "rank": data["singular_vector"]["rank"],
        }
        if load_mask:
            row["mask"] = data["singular_vector"]["metadata"]["mask"]
        if metric =="argmax_any":
            rows.append({"patch_type": "subspace", "result":data["singular_vector"][metric], **row})
            rows.append({"patch_type": "full", "result": data["full_rank"][metric], **row})
        else:
            assert label_types is not None
            for patch_type in ["full_rank", "singular_vector"]:
                res = aggregate_label_types(data, label_types, metric, patch_type, remove_op_order=remove_op_order)
                for label_type, label_acc in res.items():
                    rows.append({"patch_type": "full" if "full" in patch_type else "subspace",
                                 "label_type": label_type,
                                 "result": label_acc, **row})

    df = pd.DataFrame(rows)
    return df


def aggregate_label_types(
    data,
    label_types,
    metric:str,
    subsace_or_full:str="full_rank",
    remove_op_order: bool=False
):
    sorted_types = sorted(list(set(list(chain.from_iterable(label_types)))))

    if remove_op_order:
        # existing label types example: "op1_put" -> 2nd operation among all ops, a put op
        # converts to "put"
        label_types = [[l.split("_")[-1] for l in sample_label_types] for sample_label_types in label_types]
        sorted_types = [l.split("_")[-1] for l in sorted_types]  # still want to preserve the order of the ops

    layer_result = {t:[] for t in sorted_types}
    for sample_idx, sample_result in enumerate(data[subsace_or_full][metric]):
        # for each sample, aggregate across label types (if object is from description/put phrase)
        sample_label_types = label_types[sample_idx]
        # not each label type will appear (op0_put, or op1_put may not both appear)
        agg_sample_result = {t:0 for t in list(set(sample_label_types))}
        for obj_idx, obj_correct in enumerate(sample_result):
            obj_type = sample_label_types[obj_idx]
            # if any of the object of that op type is predicted correctly, count that operation as success
            agg_sample_result[obj_type] = agg_sample_result[obj_type] or obj_correct

        for t, res in agg_sample_result.items():
            layer_result[t].append(res)

    # now average across samples because there should be constant # of query ops
    sorted_layer_result = [np.mean(layer_result[t]) for t in sorted_types]
    return {sorted_types[i]:sorted_layer_result[i] for i in range(len(sorted_types))}


def plot_single_experiment_argmax_any(exp_dir: str):
    df = load_single_experiment(exp_dir)
    ax = sns.lineplot(x="layer", y="result", data=df, hue="patch_type")
    ax2 = ax.twinx()
    sns.scatterplot(x="layer", y="rank", data=df, label="rank", ax=ax2, color="black", marker="X")
    plt.legend()
    exp_dir_name = exp_dir.split("/")[-1]
    plt.title(exp_dir_name)

    plot_path = Path(exp_dir).parent.joinpath(f"{exp_dir_name}_argmax_any.png").resolve()
    plt.savefig(plot_path)
    plt.close()



def plot_single_experiment_full(
    exp_dir: str,
    metric: str="topk_full",
    label_types: List[List[str]]=None,
    remove_op_order: bool=False
)-> pd.DataFrame:
    assert label_types is not None
    df = load_single_experiment(exp_dir, metric=metric, label_types=label_types, remove_op_order=remove_op_order)
    metric_name = "Intervention Accuracy (Argmax)" if "argmax" in metric else "Intervention Accuracy (Top-K)"
    df.rename(columns={
        "layer":"Layer",
        "result": metric_name,
        "rank": "Rank",
        "patch_type": "Patch Type",
        "label_type":"Object Source"
    }, inplace=True)

    fix_fonts()

    ax = sns.lineplot(x="Layer", y=metric_name, data=df, style="Patch Type", hue="Object Source")
    ax2 = ax.twinx()
    sns.scatterplot(x="Layer", y="Rank", data=df, label="Rank", ax=ax2, color="black", marker="X")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Combine them into a single list
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    # Create the combined legend on one of the axes (e.g., ax1) ---
    ax.legend(all_handles, all_labels)#, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.legend()
    ax2.get_legend().remove()
    exp_dir_name = exp_dir.split("/")[-1]
    # plt.title(exp_dir_name)
    circuit_optimized_for = exp_dir.split("_")[-1].upper()
    plt.title(f"Subspace Optimized for the {circuit_optimized_for} Object")
    plt.tight_layout()

    plot_path = Path(exp_dir).parent.joinpath(f"{exp_dir_name}_{metric}{'_removeOpOrder' if remove_op_order else ''}.png").resolve()
    plt.savefig(plot_path, dpi=600)
    plt.savefig(str(plot_path).replace(".png", ".pdf"), dpi=600)
    plt.close()


def subspace_patch_main(args):
    if args.remote:
        setup_nnsight()
        
    dataloader, dataset, model = get_model_and_dataset(args)
    train_dataloader, valid_dataloader = split_train_validation_data(dataset, model.tokenizer, args)
    os.makedirs(args.output_dir, exist_ok=True)
    singular_vectors, _ = get_basis_directions(model, args, cache_dir=args.basis_direction_cache_path, remote=args.remote)
    exp_name = args.counterfactual_format.replace("dcm_", "")
    for layer in tqdm(range(model.config.num_hidden_layers)):
    # for layer in tqdm([20]):
        layer_output_path = os.path.join(args.output_dir, f"{layer}.json")
        if os.path.exists(layer_output_path):
            if args.verbose:
                print(f"Skipping {layer_output_path} as results exists.")
            continue

        layer_performance = {}
        # full state patching
        full_argmax_any, full_argmax_full, full_topk_full = validate(
            exp_name,
            model,
            layer,
            valid_dataloader,
            verbose=args.verbose,
            remote=args.remote,
        )
        print("-" * 30)
        print(f"Full state patching val: {full_argmax_any}")
        print("-" * 30)
        layer_performance["full_rank"] = {"argmax_any": full_argmax_any, "rank": None, "argmax_full": full_argmax_full, "topk_full": full_topk_full}

        # singular vector patching
        training_metadata = {
            "learning_rate": args.learning_rate,
            "n_epochs": args.n_epochs,
            "lamb": args.lamb,
        }
        print(f"Training singular vectors with {training_metadata}")

        singular_projection, singular_metadata = get_low_rank_projection(
            exp_name=exp_name,
            lm=model,
            layer_idx=layer,
            train_loader=train_dataloader,
            basis_directions=singular_vectors[layer],
            learning_rate=args.learning_rate,
            n_epochs=args.n_epochs,
            lamb=args.lamb,
            verbose=args.verbose,
            target_object_types=args.target_object_types,
            remote=args.remote,
        )

        print("validating ...")
        singular_argmax_any, singular_argmax_full, singular_topk_full = validate(
            exp_name=exp_name,
            lm=model,
            layer_idx=layer,
            validation_loader=valid_dataloader,
            projection=singular_projection,
            verbose=args.verbose,
            remote=args.remote,
        )
        print("-" * 30)
        print(f"Singular vector patching val: {singular_argmax_any} | Rank: {singular_metadata['rank']}")
        print("-" * 30)

        layer_performance["singular_vector"] = {
            "argmax_any": singular_argmax_any,
            "rank": singular_metadata["rank"],
            "metadata": {
                "training_args": training_metadata,
                "mask": singular_metadata["mask"],
            },
            "argmax_full": singular_argmax_full,
            "topk_full": singular_topk_full,
        }

        # save results after each layer
        with open(layer_output_path, "w") as f:
            json.dump(layer_performance, f, indent=4)

    # visualize the experiment results
    # plot_single_experiment_argmax_any(args.output_dir)
    if len(args.query_ops_order) > 0:
        for remove_op_order in [True]: # , False
            plot_single_experiment_full(args.output_dir, metric="topk_full", label_types=valid_dataloader.dataset["source_label_types"], remove_op_order=remove_op_order)
            plot_single_experiment_full(args.output_dir, metric="argmax_full", label_types=valid_dataloader.dataset["source_label_types"], remove_op_order=remove_op_order)
    return


def subspace_patch_plot_only(args):
    if args.remote:
        setup_nnsight()

    dataloader, dataset, model = get_model_and_dataset(args)
    train_dataloader, valid_dataloader = split_train_validation_data(dataset, model.tokenizer, args)
    os.makedirs(args.output_dir, exist_ok=True)
    singular_vectors, _ = get_basis_directions(model, args, cache_dir=args.basis_direction_cache_path,
                                               remote=args.remote)

    # visualize the experiment results
    plot_single_experiment_argmax_any(args.output_dir)
    if len(args.query_ops_order) > 0:
        for remove_op_order in [True]: # , False
            plot_single_experiment_full(args.output_dir, metric="topk_full", label_types=valid_dataloader.dataset["source_label_types"], remove_op_order=remove_op_order)
            plot_single_experiment_full(args.output_dir, metric="argmax_full", label_types=valid_dataloader.dataset["source_label_types"], remove_op_order=remove_op_order)
    return

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--train_size', help='number of training data', type=int, default=80)
    parser.add_argument('--validation_size', help='number of validation data', type=int, default=80)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.1)
    parser.add_argument('--train_batch_size', help='batch size of training data', type=int, default=8)
    parser.add_argument('--n_epochs', help='number of epochs to learn subspace masking', type=int, default=1)
    parser.add_argument('--lamb', help='L1 regularization parameter', type=float, default=0.1)
    parser.add_argument('--basis_direction_cache_path', help='path to cache basis directions', type=str, default="../outputs/nnsight_patch_noop/gemma-2-2b")
    parser.add_argument('--verbose', help='verbose level', action='store_true')
    parser.add_argument('--target_object_types', help='comma separated list of object types to use as loss', type=str, default=None)
    return parser

def more_fix_args(args):
    if args.target_object_types is not None:
        args.target_object_types = args.target_object_types.split(",")

if __name__ == "__main__":
    parser = add_args(build_parser())
    args = parser.parse_args()
    print(f"ARGS: {args}")
    post_arg_parse_fix(args)
    more_fix_args(args)
    fix_random_seed(args.seed)
    subspace_patch_main(args)
    # subspace_patch_plot_only(args)