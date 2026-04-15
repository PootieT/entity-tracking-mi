import os
import pdb
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import argparse

import sys
import torch
import time 
from tqdm import tqdm
import numpy as np

import nnsight
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from nnsight import CONFIG, LanguageModel
import os

sys.path.append("..")
from utils import get_model_and_tokenizer, fix_random_seed, get_random_circuit, get_circuit, eval_circuit_performance, \
    MODEL_TO_SHORT, load_dataloader, get_objects
sys.path.append("../nnsight_patching_experiment")
from patch_utils import build_parser, post_arg_parse_fix, maybe_patch_or_load_cache, maybe_logit_soft_capping, setup_nnsight



def run_behavioral_test(model, dataloader, args) -> List[float]:
    avg_accuracies = []
    n_instances = 0
    for batch in dataloader:
        for toks, labels, last_token_idx in zip(batch["base_tokens"], batch["labels"], batch["base_last_token_indices"]):
            prompt = model.tokenizer.decode(toks, skip_special_tokens=True)
            n_corrects = 0
            already_predicted = set()

            # We will perform #labels predictions
            # NOTE: it should not be a problem if we're only running 100ish points, but if we want to scale up we should consider batching these to a session, which might be tricky, so just leave it for now.
            for _ in range(len(labels)):
                with torch.no_grad(), model.trace(prompt, remote=args.use_ndif_remote) as tracer:
                    logits = model.lm_head.output.save()
                logits = maybe_logit_soft_capping(logits, model)

                # Greedy decoding
                predicted_token = logits[0, last_token_idx, :].argmax().item()

                # Check if the prediction is correct and if it's the first occurrence
                if predicted_token in labels and not predicted_token in already_predicted:
                    n_corrects += 1
                
                # Update the already predicted tokens
                already_predicted.add(predicted_token)

                # Appending the predicted token to the prompt, alongside a ,
                prompt += model.tokenizer.decode(predicted_token, skip_special_tokens=True)
                prompt += ","
                # if using ndif remote, sleep 20s to avoid rate limit
                if args.use_ndif_remote:
                    time.sleep(20)

            assert 0 <= n_corrects <= len(labels)

            avg_accuracies.append(n_corrects / len(labels))
            n_instances += 1

    return avg_accuracies


def get_max_possible_tokens(labels: List):
    """Return how many tokens we want to generate at most
    ex1: ... contains the bla and the bla and the bla -> 3 X # labels
    ex2: ... contains the bla, the bla, and the bla -> < 3 X # labels"""
    return 10 * len(labels)


def inference_or_generate_batch(
    args: argparse.Namespace,
    batch: Dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inference_only: bool,
    use_ctf: bool=False,
    label_field:Optional[str]=None
):
    """
    forward model 1 or more times and collect output in the form of generation or logit

    Args:
        args (argparse.Namespace): experiment args
        batch (List[Any]): batch of inputs

    """
    # pdb.set_trace()
    batch_size = len(batch["labels"])
    field_prefix = "base_" if not use_ctf else "source_"
    if label_field is None:
        label_field = "labels" if not use_ctf else "source_labels"
    sampling_kwargs = {} if args.sampling == "greedy" else {"do_sample": True}
    if not inference_only:
        if len(np.unique(batch[f"{field_prefix}last_token_indices"])) == 1:  # if all data of the same batch has same length can decode by batch
            max_batch_gen_tokens = max(get_max_possible_tokens(l) for l in batch[label_field])
            if not args.use_ndif_remote:
                batch[f"{field_prefix}tokens"] = batch[f"{field_prefix}tokens"].to(model.device)
                out = model.generate(batch[f"{field_prefix}tokens"], max_new_tokens=max_batch_gen_tokens, **sampling_kwargs) #attention_mask=attention_mask, 
            else:
                batch[f"{field_prefix}tokens"] = batch[f"{field_prefix}tokens"].to("cpu")
                with model.generate(batch[f"{field_prefix}tokens"], max_new_tokens=max_batch_gen_tokens, remote=True, **sampling_kwargs) as gen_tracer:
                    out = model.generator.output.save()
                

        else:
            # this is problematic
            out = []
            # maybe not the most effective solution, best to sort data by length to optimizing batching
            inputs = []
            for i in range(batch_size):
                # convert to string and let ndif handle the padding
                str_input = model.tokenizer.decode(batch[f"{field_prefix}tokens"][i], skip_special_tokens=True)
                inputs.append(str_input)
            if args.use_ndif_remote:
                with model.generate(inputs, remote=True, **sampling_kwargs) as gen_tracer:
                    all_outputs = model.generator.output.save()
                for i in range(batch_size):
                    max_gen_tokens = get_max_possible_tokens(batch[label_field][i])
                    out.append(all_outputs[i])
            else:
                for i in range(batch_size):
                    max_gen_tokens = get_max_possible_tokens(batch[label_field][i])
                    no_pad_toks = tokenizer.encode(tokenizer.decode(batch[f"{field_prefix}tokens"][i], skip_special_tokens=True),return_tensors="pt")
                    output = model.generate(no_pad_toks, max_new_tokens=args.fix_new_tokens, **sampling_kwargs)[0]
                    out.append(output)
    else:  # just forward pass/inference
        # pdb.set_trace()
        if not args.use_ndif_remote:
            batch[f"{field_prefix}tokens"] = batch[f"{field_prefix}tokens"].to(model.device)
        if args.use_ndif_remote:
            
            with model.trace(batch[f"{field_prefix}tokens"], remote=True) as tracer:
                out = model.output['logits'].detach().cpu().save()
        else:
            out = model(batch[f"{field_prefix}tokens"])["logits"] # not compatible with nnsight
    return out


def run_behavioral_test_unconstrained(model, tokenizer, dataloader, args) ->Dict:
    with torch.no_grad():
        rows = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch_size = len(batch["labels"])
            # pdb.set_trace()
            out = inference_or_generate_batch(args, batch, model, tokenizer, inference_only="logit" in args.metric, label_field="source_labels" if "diff" in args.metric else None)
            ctf_out = None
            if args.metric == "logit_diff":
                ctf_out = inference_or_generate_batch(args, batch, model, tokenizer, inference_only=True, use_ctf=True)
            for i in range(batch_size):
                row = {"dataset_index": int(batch["dataset_indices"][i])}
                label_texts = [tokenizer.decode(l).strip().lower() for l in batch["labels"][i]]
                label_objs = set(label_texts)
                last_token_idx = batch["base_last_token_indices"][i] if not args.fix_new_tokens else -args.fix_new_tokens
                if args.metric == "accuracy":
                    full_answer = tokenizer.decode(out[i][last_token_idx:].cpu()).lower()
                    decoded_answer = full_answer.split(".")[0].split("box")[0]
                    decoded_objs = set(get_objects(decoded_answer))
                    row["recall"] = np.mean([1 if l in decoded_answer else 0 for l in label_texts])
                    row["precision"] = np.mean([1 if o in label_texts else 0 for o in decoded_objs])
                    row["decoded_answer"] = decoded_answer
                    row["full_decoded_answer"] = full_answer
                    res = label_objs == decoded_objs
                elif args.metric == "logit_diff":
                    # pdb.set_trace("entering logit diff")
                    labels_ids = batch["source_labels"][i].tolist()
                    logit = out[i,last_token_idx,labels_ids].mean().cpu()
                    ctf_last_token_idx = batch["source_last_token_indices"][i]
                    ctf_logit = ctf_out[i,ctf_last_token_idx,labels_ids].mean().cpu()
                    res = (ctf_logit - logit).item()

                    row["argmax_token"] = tokenizer.decode(out[i,last_token_idx].argmax())
                    row["ctf_argmax_token"] = tokenizer.decode(ctf_out[i,ctf_last_token_idx].argmax())
                    row["ctf_label"] = [tokenizer.decode(l).strip().lower() for l in batch["source_labels"][i]]
                    row["ctf_sentence"] = tokenizer.decode(batch["source_tokens"][i], skip_special_tokens=True)
                    row["sentence"] = tokenizer.decode(batch["base_tokens"][i], skip_special_tokens=True)

                    # Additionally save the rank of the objects
                    # rank_diff =
                    # = (rank_ctf_obj - rank_ctf_target) - (rank_noop_obj - rank_noop_target)
                    # = (rank_ctf_obj - rank_noop_obj) - (rank_ctf_target - rank_noop_target)

                    assert len(labels_ids) == 1, "for ranking, it's less sensible to average, comment this out if you think otherwise"
                    rank = torch.argsort(torch.argsort(out[i,last_token_idx], descending=True, dim=-1), dim=-1) + 1
                    ctf_rank = torch.argsort(torch.argsort(ctf_out[i,ctf_last_token_idx], descending=True, dim=-1), dim=-1) + 1
                    row["rank_diff"] = (ctf_rank[labels_ids][0] - rank[labels_ids][0]).cpu().numpy().tolist()

                    if args.compute_other_object_logit_diffs:
                        # get logit diff of the target object of counterfactual
                        base_labels = [l for l in batch["labels"][i].tolist() if l not in batch["source_labels"][i].tolist()]
                        base_label_res = (ctf_out[i,ctf_last_token_idx,base_labels] - out[i,last_token_idx,base_labels]).cpu().numpy().tolist()
                        row[f"target_objs"] = [tokenizer.decode(l).strip().lower() for l in base_labels]
                        row[f"target_objs_{args.metric}"] = base_label_res
                        row[f"target_objs_rank_diff"] = (ctf_rank[base_labels] - rank[base_labels]).cpu().numpy().tolist()
                        row[f"target_objs_rank"] = rank[base_labels].cpu().numpy().tolist()
                        row[f"target_objs_ctf_rank"] = ctf_rank[base_labels].cpu().numpy().tolist()

                        # get logit diff of all other objects
                        sentence_no_shot = row["sentence"] if not args.prompt_format else row["sentence"].split("\n\nDescription: ")[-1].split("\nStatement: ")[0]
                        other_objects = [o for o in get_objects(sentence_no_shot) if ((o not in row[f"target_objs"]) and (o not in row["ctf_label"]))]
                        other_labels = [tokenizer.encode(f"the {o}")[-1] for o in other_objects]
                        other_obj_res = (ctf_out[i,ctf_last_token_idx,other_labels] - out[i,last_token_idx,other_labels]).cpu().numpy().tolist()
                        row["other_objs"] = [tokenizer.decode(l).strip().lower() for l in other_labels]
                        row[f"other_objs_{args.metric}"] = other_obj_res
                        row[f"other_objs_rank_diff"] = (ctf_rank[other_labels] - rank[other_labels]).cpu().numpy().tolist()

                else:
                    raise NotImplementedError
                row[args.metric] = res
                if args.save_data:
                    row["labels"] = label_texts
                    row["sentence"] = tokenizer.decode(batch["base_tokens"][i], skip_special_tokens=True)
                rows.append(row)

    avg = np.mean([row[args.metric] for row in rows])
    print(f"Average metric: {avg}")
    return {f"avg_{args.metric}": avg, "full_results": rows}



def behavioral_test_main(args: argparse.Namespace):
    """
    Run behavioral testing for models on specific dataset, script is generic to accomadate
    different experiments, sampling parameters, metric, etc.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        quantization_config = None
    if not args.use_ndif_remote:
        model =  AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization_config)
    else:
        model = LanguageModel(args.model, device_map="auto")

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    if any([t in args.model for t in ["gemma", "Llama-3.", "santacoder", "gpt2", "Qwen"]]):
        prepend_space_to_answer = True
    else:
        prepend_space_to_answer = False
    print("MODEL LOADED")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.load_in_8bit and not args.load_in_4bit and not args.use_ndif_remote:
        model = model.to(device) 
  
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # load dataset
    dataloader, dataset = load_dataloader(
        model=model,
        tokenizer=tokenizer,
        datafile=args.datafile,
        num_samples=args.num_samples,
        num_boxes=7,  # args.num_boxes,
        ops_order=args.ops_order,
        query_ops_order=args.query_ops_order,
        success_filter=args.success_filter,
        put_globally_removed_filter=args.put_globally_removed_filter,
        operations_on_same_obj=args.operations_on_same_obj,
        copy_filter=args.copy_filter,
        batch_size=args.batch_size,
        return_dataset=True,
        max_initial_objects_per_box=args.max_initial_objects_per_box,
        counterfactual_format=args.counterfactual_format,
        data_field=args.data_field,
        token_step=args.token_step,
        prepend_space_to_answer=prepend_space_to_answer,
        num_query_object=args.num_query_object,
        sort_query_objects=args.sort_query_objects,
        prompt_format=args.prompt_format,
        seed=args.seed,
        remote=args.use_ndif_remote,
    )
    print(f"DATALOADER CREATED ({len(dataset)=})")
    print(f"max data length: {len(dataset['base_tokens'][0])}")
    dirs = "/".join(args.output_dir.split("/")[:-1])
    os.makedirs(dirs, exist_ok=True)

    results = maybe_patch_or_load_cache(
        f"{args.output_dir}",
        run_behavioral_test_unconstrained,
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        args=args
    )
    print(f"Average {args.metric} = {results['avg_'+args.metric]}")
    if args.save_data:
        # convert it to probing dataset format so we can use it later for probe intervention
        formatted_data = []
        for row in results["full_results"]:
            query_box = int(row["sentence"].split(" ")[-3])
            numops = sum([1 if f"Box {query_box}" in op else 0 for op in row["sentence"].split(". ")[1:-1]])
            
            # a patchy way to resolve missing decoded answer
            # if "decoded_answer" not in row or row["decoded_answer"] is None:
            #     row["decoded_answer"] = "Not Given(probably because no acc exp yet)"
            new_row = {
                "prefix": row["sentence"].removesuffix(" the"),
                "original_answer": row["decoded_answer"],
                "parsed_original_answer": get_objects(row["decoded_answer"]),
                "gold_items": row["labels"],
                "gold_answer": " and ".join([f"the {obj}" for obj in row["labels"]]),
                "numops_global": row["sentence"].count(". ") - 1,
                "numops": numops,
                "correct": row.get("accuracy"),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
            }
            formatted_data.append(new_row)
        with open(args.output_dir.replace(".json", ".jsonl"), "w") as f:
            f.writelines([json.dumps(row)+"\n" for row in formatted_data])



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "first_token_argmax_any", "logit_diff"])
    parser.add_argument("--sampling", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--save_data", action="store_true", help="whether to save input data (sentence, labels, etc)")
    parser.add_argument("--put_globally_removed_filter", type=str2bool, default=None)
    parser.add_argument("--compute_other_object_logit_diffs", action="store_true", help="whether to compute logit diff for all other objects")
    parser.add_argument("--use_ndif_remote", action="store_true", help="whether to use ndif remote model")
    parser.add_argument("--fix_new_tokens", type=int)
    return parser


if __name__ == "__main__":
    parser = add_args(build_parser())
    args = parser.parse_args()
    post_arg_parse_fix(args)
    print(f"ARGS: {args}")
    fix_random_seed(args.seed)
    if args.use_ndif_remote:
        setup_nnsight()
    pdb.set_trace()
    behavioral_test_main(args)
