import os
import pdb
from itertools import chain

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import nnsight
from nnsight import CONFIG, LanguageModel, util
import pandas as pd  # this has to be after nnsight or throws gcc error

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("..")
from utils import get_model_and_tokenizer, fix_random_seed, get_random_circuit, get_circuit, eval_circuit_performance, MODEL_TO_SHORT, force_pad, check_prompt_success, fix_fonts, setup_nnsight
from patch_utils import build_parser, post_arg_parse_fix, get_model_and_dataset, maybe_patch_or_load_cache


def activation_patching_residual_stream_old(
    model: LanguageModel,
    clean_tokens: np.ndarray,
    corrupted_tokens: np.ndarray,
    label_tokens: List[List[int]],
    last_token_pos: List[int],
    args: argparse.Namespace,
):
    N_LAYERS = model.config.num_hidden_layers
    N_HEADS = model.config.num_attention_heads
    N_DATA = len(clean_tokens)

    # ======== Step 1 ==========
    # gather activations for clean and corrupt run
    # Clean run (breaking into multiple tracer calls because otherwise we run into OOM)
    clean_hs, clean_logits, clean_probs = cache_logit_and_hidden(model=model, batch_size=args.batch_size,
                                                                 tokens_ids=clean_tokens,
                                                                 last_token_pos=last_token_pos,
                                                                 label_indices=label_tokens, save_hs=True,
                                                                 reshape=False, module="resid")
    # Corrupted run
    _, corrupted_logits, corrupted_probs = cache_logit_and_hidden(model=model, batch_size=args.batch_size,
                                                                  tokens_ids=corrupted_tokens,
                                                                  last_token_pos=last_token_pos,
                                                                  label_indices=label_tokens, module="resid")
    N_TOKENS = clean_hs[0].shape[1]
    # Activation Patching Intervention
    patching_results = []
    # Iterate through all the layers
    bar = tqdm(total=N_LAYERS * N_TOKENS)
    for layer_idx in range(N_LAYERS):
        _patching_results = []
        # Iterate through all tokens
        for token_idx in range(N_TOKENS):
            # iterate through batches
            patched_result_sum = torch.zeros(1)
            for batch_i in range(0, N_DATA, args.batch_size):
                batch_indices = range(batch_i, min(N_DATA, batch_i + args.batch_size))
                batch_corrupted_tokens = force_pad(corrupted_tokens[batch_indices], model.tokenizer)

                # Patching corrupted run at given layer and token
                torch.cuda.empty_cache()
                with torch.no_grad():
                    with model.trace(batch_corrupted_tokens) as tracer:
                        # Apply the patch from the clean hidden states to the corrupted hidden states.
                        model.model.layers[layer_idx].self_attn.o_proj.input[:, token_idx, :] = torch.stack(
                            [clean_hs[b] for b in batch_indices])[:, layer_idx, token_idx, :]
                        patched_logits = model.lm_head.output
                        patched_logits = maybe_logit_soft_capping(patched_logits, model).save()
                        patched_probs = torch.softmax(patched_logits, dim=-1).save()

                    patched_logits_batch = [
                        patched_logits[bi, last_token_pos[batch_indices[bi]], label_tokens[batch_indices[bi]]] for
                        bi in range(len(batch_indices))]
                    patched_probs_batch = [
                        patched_probs[bi, last_token_pos[batch_indices[bi]], label_tokens[batch_indices[bi]]] for bi
                        in range(len(batch_indices))]

                    # Calculate the improvement in the correct token after patching.
                    batch_patched_result = get_patch_score(patched_logits_batch, patched_probs_batch,
                                                           [clean_logits[bi] for bi in batch_indices],
                                                           [clean_probs[bi] for bi in batch_indices],
                                                           args.use_object_index, args.score_source == "prob",
                                                           [corrupted_logits[bi] for bi in batch_indices],
                                                           [corrupted_probs[bi] for bi in batch_indices],
                                                           )
                    for bi in range(len(batch_indices)):
                        patched_result_sum = (patched_result_sum + batch_patched_result[bi].cpu())

            patch_result_avg = patched_result_sum / N_DATA
            _patching_results.append(patch_result_avg)
            bar.update(1)
        patching_results.append(_patching_results)

    for i in range(min(len(clean_tokens), 2)):
        print(f"example {i}")
        print(f"Clean logit: {clean_logits[i].tolist()}, clean prob: {clean_probs[i].tolist()}")
        print(f"Corrupted logit: {corrupted_logits[i].tolist()}, corrupted prob: {corrupted_probs[i].tolist()}")

    return patching_results



def activation_patching_residual_stream(
    model: LanguageModel,
    dataset: Dataset,
    args: argparse.Namespace,
):
    """
    patch from clean token position residual stream to respective counterfactual token positions, and measure
    accuracy of expected target objects
    """
    clean_tokens = dataset["base_tokens"]
    corrupted_tokens = dataset["source_tokens"]
    label_tokens = list(dataset["source_labels"])

    if dataset["patch_locations"][0] is None:
        clean_pos = list(dataset["base_last_token_indices"])
        corrupted_pos = list(dataset["source_last_token_indices"])
    else:
        # format is List[List[Tuple[int, int]]]: [[(src, tgt), (src, tgt)], [], ...]
        patch_pos = list(dataset["patch_locations"])
        clean_pos = [[tup[1] for tup in data_pos] for data_pos in patch_pos]
        corrupted_pos = [[tup[0] for tup in data_pos] for data_pos in patch_pos]
        print(f"patching at {len(patch_pos[0])} token positions")
    last_token_pos = np.array(dataset["base_last_token_indices"])

    N_LAYERS = model.config.num_hidden_layers
    N_HEADS = model.config.num_attention_heads
    N_DATA = len(clean_tokens)
    MAX_N_LABELS = 5 if args.debug else max(len(l) for l in label_tokens)

    argmax_correct_any = []
    argmax_correct_full = []
    topk_correct_full = []

    # Iterate through all the layers
    bar = tqdm(range(21, N_LAYERS)) if args.debug else tqdm(range(N_LAYERS))
    for layer_idx in bar:

        _argmax_correct_any = 0
        _argmax_correct_full = []
        _topk_correct_full = []

        # iterate through batches
        for batch_i in range(0, N_DATA, args.batch_size):
            batch_indices = range(batch_i, min(N_DATA, batch_i + args.batch_size))
            batch_corrupted_tokens = force_pad(corrupted_tokens[batch_indices], model.tokenizer)
            batch_clean_tokens = force_pad(clean_tokens[batch_indices], model.tokenizer)
            batch_clean_token_pos = [clean_pos[bi] for bi in batch_indices]
            batch_corrupted_token_pos = [corrupted_pos[bi] for bi in batch_indices]

            # Patching corrupted run at given layer and token
            torch.cuda.empty_cache()

            patch_layers = [layer_idx] if args.patch_style == "single_layer" else list(range(layer_idx)) if args.patch_style=="first_n" else list(range(layer_idx, N_LAYERS))

            with torch.no_grad():
                corrupt_layer_outs = {}
                with model.trace(remote=args.remote) as tracer:
                    with tracer.invoke(batch_corrupted_tokens):
                        for patch_layer in patch_layers:
                            corrupt_layer_outs[patch_layer] = model.model.layers[patch_layer].output[0][:,batch_corrupted_token_pos].clone().save()
                            # corrupt_layer_outs[patch_layer] = model.model.layers[patch_layer].output[:,batch_corrupted_token_pos].clone().save()
                            print(f"length of cache: {len(corrupt_layer_outs)}")
                    # patch into clean run

                    with tracer.invoke(batch_clean_tokens):
                        for patch_layer in patch_layers:
                            # somehow running into error where corrupt_layer_outs is not storing the activations (empty)
                            model.model.layers[patch_layer].output[0][:, batch_clean_token_pos] = corrupt_layer_outs[patch_layer]
                            # model.model.layers[patch_layer].output[:, batch_clean_token_pos] = corrupt_layer_outs[patch_layer]
                        logits = model.lm_head.output
                        last_token_logits = logits[range(len(batch_indices)), last_token_pos[batch_indices]]
                        topk_pred = last_token_logits.argsort(dim=-1, descending=True)[:,:MAX_N_LABELS].cpu().numpy().save()

            for i in range(len(batch_indices)):
                labels = label_tokens[batch_indices[i]]  # multiple target objects
                label_texts = [model.tokenizer.decode(l).strip().lower() for l in labels]
                topk_pred_texts = [model.tokenizer.decode(l).strip().lower() for l in topk_pred[i, :len(label_texts)]]
                if args.debug:
                    print(f"Corrupted Sentence: {model.tokenizer.decode(batch_corrupted_tokens[i])}")
                    print(f"Clean     Sentence: {model.tokenizer.decode(batch_clean_tokens[i])}")
                    topk_five_texts = [model.tokenizer.decode(l).strip().lower() for l in topk_pred[i, :5]]
                    print(f"Expected Labels: {label_texts}")
                    print(f"Top 5 prediction: {topk_five_texts}")

                if topk_pred_texts[0] in label_texts:
                    _argmax_correct_any+=1

                argmax_correct_full_batch = []
                topk_correct_full_batch = []
                for k, label_text in enumerate(label_texts):
                    argmax_correct_full_batch.append(1 if label_text == topk_pred_texts[0] else 0)
                    topk_correct_full_batch.append(1 if label_text in topk_pred_texts else 0)

                _argmax_correct_full.append(argmax_correct_full_batch)
                _topk_correct_full.append(topk_correct_full_batch)
        if args.debug:
            pdb.set_trace()
        argmax_correct_any.append(_argmax_correct_any/N_DATA)
        argmax_correct_full.append(_argmax_correct_full)
        topk_correct_full.append(_topk_correct_full)
        bar.set_description(f"L{layer_idx} argmax any={_argmax_correct_any/N_DATA:.3f}")
        
    return argmax_correct_any, argmax_correct_full, topk_correct_full
    

def plot_accuracy(results: List[List[Union[float, List[float]]]], out_path: str, multi_target_treatment: str="front_fill", label_types: Optional[List[List[str]]]=None):
    """
    Results will be of shape n_layer X n_samples X n_targets (where n_target could vary)

    Args:
        results (List[List[Union[float, List[float]]]]): activation patching results
        out_path (str): output path
        multi_target_treatment (str):
            - "sum": sum the probabilities for all objects into a single number
            - "front_fill": fill less target datapoints from the front (1st object)
            - "back_fill": fill less target datapoints from the back (last object)
            - "sum_per_type": some probabilities of objects in the same phrases
        label_types (Optional[List[List[str]]]): list of label types per label. Required if multi_target_treatment=sum_per_type
    """
    try:
        # if n_target is the same across samples
        results = np.array(results)
        if results.ndim > 2:
            results.mean(1)
        col_names = [f"Obj {i}" for i in range(results.shape[1])] if results.ndim > 1 else ["Any Objs"]
        row_names = [f"Layer {i}" for i in range(results.shape[0])]
        df = pd.DataFrame(results, index=row_names, columns=col_names).melt(var_name="Object Index", value_name="Accuracy")
        df = df.reset_index().rename(columns={"index": "Layer"})
    except:
        # if n_target is different across samples
        if multi_target_treatment == "sum":
            agg_results = []
            for layer_idx in range(len(results)):
                layer_result = [np.any(s).astype(int) for s in results[layer_idx]]
                agg_results.append(layer_result)
            results = np.array(agg_results).mean(axis=1)
            col_names = ["Any Objs"]
            row_names = [f"Layer {i}" for i in range(results.shape[0])]
            df = pd.DataFrame(results, index=row_names, columns=col_names).melt(var_name="Object Index", value_name="Accuracy")
            df = df.reset_index().rename(columns={"index": "Layer"})
        elif multi_target_treatment in {"front_fill", "back_fill"}:
            agg_results = []
            max_targets = max(len(s) for s in results[0])
            for layer_idx, layer_result in enumerate(results):
                for sample_idx, sample_result in enumerate(layer_result):
                    sample_total_objects = len(sample_result)
                    backfill_offset = max_targets-sample_total_objects
                    for obj_idx, obj_correct in enumerate(sample_result):
                        if multi_target_treatment == "front_fill":
                            agg_results.append({"Layer": layer_idx, "Object Index": obj_idx, "Accuracy": obj_correct})
                        elif multi_target_treatment == "back_fill":
                            agg_results.append({"Layer": layer_idx, "Object Index": obj_idx + backfill_offset, "Accuracy": obj_correct})
            df = pd.DataFrame(agg_results)
        elif multi_target_treatment == "sum_per_type":
            assert label_types is not None
            agg_results = []
            # sorted_types = sorted(list(set(label_types[0]))) # sometimes order of query op is not deterministic
            sorted_types = sorted(list(set(list(chain.from_iterable(label_types)))))
            type_index = {t: i for i, t in enumerate(sorted_types)}
            for layer_idx in range(len(results)):
                layer_result = []
                for sample_idx, sample_result in enumerate(results[layer_idx]):
                    # for each sample, aggregate across types
                    sample_label_types = label_types[sample_idx]
                    agg_sample_result = [0]*len(sorted_types)
                    for obj_idx, obj_correct in enumerate(sample_result):
                        obj_type_idx = type_index[sample_label_types[obj_idx]]
                        agg_sample_result[obj_type_idx] += obj_correct
                    # now count correct if any of the same type obj is correct
                    for i, t in enumerate(agg_sample_result):
                        agg_sample_result[i] = int(agg_sample_result[i] > 0)
                    layer_result.append(agg_sample_result)
                # now average across samples because there should be constant # of query ops
                layer_result = np.array(layer_result)# .mean(axis=0)
                agg_results.append(layer_result)
            agg_results = np.array(agg_results)
            dfs = []
            col_names = sorted_types
            row_names = range(agg_results.shape[0])
            for sample_idx in range(agg_results.shape[1]):
                df = pd.DataFrame(agg_results[:, sample_idx], index=row_names, columns=col_names).melt(ignore_index=False, var_name="Object Operation Type", value_name="Accuracy")
                df = df.reset_index().rename(columns={"index": "Layer"})
                dfs.append(df)
            df = pd.concat(dfs)
        else:
            raise NotImplementedError
    fix_fonts()
    # sns.set_theme(font_scale=1.5)  # customize for poster
    hue_col = "Object Operation Type" if multi_target_treatment == "sum_per_type" else "Object Index"
    ax = sns.lineplot(df, x="Layer", y="Accuracy", hue=hue_col)
    ax.set_ylabel("Intervention Accuracy")
    plt.tight_layout()
    ###########
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".png", ".pdf"), dpi=600)
    plt.close()
    return


def plot_accuracy_by_id(model, dataset, results, out_path):
    # plot by box id
    results = np.array(results).squeeze()
    prompts = model.tokenizer.batch_decode(dataset['base_tokens'], skip_special_tokens=True)
    query_id = np.array([int(p.split()[-3]) for p in prompts])
    acc_by_id = [results[:, query_id == i] for i in range(7)]
    for i in range(7):
        print(f"Box {i}, n={acc_by_id[i].shape[1]}, max acc across layers={acc_by_id[i].mean(1).max(0)}")

    num_layers, num_samples = results.shape

    df = pd.DataFrame(results, index=range(num_layers))
    df = df.reset_index().melt(id_vars='index', var_name='sample', value_name='accuracy')
    df = df.rename(columns={'index': 'layer'})

    # Add box_id info
    df['box_id'] = np.tile(query_id, num_layers)

    # Plot
    plt.figure()
    sns.lineplot(data=df, x='layer', y='accuracy', hue='box_id', estimator='mean', ci='sd')
    plt.title("Accuracy across layers by query Box ID")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def hypothesis_patch_main(args: argparse.Namespace):
    """
    activation patch residual stream, measure accuracy of expected hypothesis supported target
    """
    if args.remote:
        setup_nnsight()

    dataloader, dataset, model = get_model_and_dataset(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Ctf:   {model.tokenizer.decode(dataset['source_tokens'][0])}")
    print(f"Input: {model.tokenizer.decode(dataset['base_tokens'][0])}")
    print(f"Labels: {[model.tokenizer.decode(t) for t in dataset['source_labels'][0]]}")

    if args.filter_ctf_success:
        success_indices = []
        for i in tqdm(range(len(dataset['source_labels']))):
            _success = check_prompt_success(
                            model,
                            tokenizer=model.tokenizer,
                            label_tokens=dataset['source_labels'][i],
                            prompt=model.tokenizer.decode(dataset['source_tokens'][i], skip_special_tokens=True),
                        )
            if _success:
                success_indices.append(i)
        print(f"ctf success rate: {len(success_indices)}/{len(dataset)} ({len(success_indices)/len(dataset)})")
        dataset = dataset.select(success_indices)
        print(f"filtered dataset size: {len(dataset)}")

    results = maybe_patch_or_load_cache(
        f"{args.output_dir}/results.pkl",
        activation_patching_residual_stream,
        model=model,
        dataset=dataset,
        args=args,
    )
    argmax_correct_any, argmax_correct_full, topk_correct_full = results

    plot_accuracy(argmax_correct_any, f"{args.output_dir}/argmax_correct_any.png")
    print(f"Argmax Correct:\n{argmax_correct_any}")
    try:
        plot_accuracy(topk_correct_full, f"{args.output_dir}/topk_correct_full_front_fill.png", "front_fill")
        plot_accuracy(argmax_correct_full, f"{args.output_dir}/argmax_correct_full_front_fill.png", "front_fill")
        plot_accuracy_by_id(model,dataset, topk_correct_full, f"{args.output_dir}/topk_correct_full_by_query_id.png")
        plot_accuracy_by_id(model, dataset, argmax_correct_full, f"{args.output_dir}/argmax_correct_full_by_query_id.png")
    except Exception as e:  # labels have different numbers of objects
        for option in ["sum", "sum_per_type", "front_fill", "back_fill",]: # "sum_per_type"
            plot_accuracy(topk_correct_full, f"{args.output_dir}/topk_correct_full_{option}.png", option, label_types=list(dataset["source_label_types"]))
            plot_accuracy(argmax_correct_full, f"{args.output_dir}/argmax_correct_full_{option}.png", option, label_types=list(dataset["source_label_types"]))



def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--filter_ctf_success", action="store_true")
    parser.add_argument("--patch_style", type=str, default="single_layer",
                        choices=["single_layer", "first_n", "last_n"])
    return parser

if __name__ == "__main__":
    parser = add_args(build_parser())
    args = parser.parse_args()
    print(f"ARGS: {args}")
    post_arg_parse_fix(args)
    fix_random_seed(args.seed)
    hypothesis_patch_main(args)
