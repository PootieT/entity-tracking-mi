import re
import os
import csv
import pdb
import json
import argparse
from typing import AnyStr, List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict
import anyio
import h5py
import scipy
import torch
import numpy as np
import pandas as pd
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer
import nnsight
from tqdm import tqdm
import time

from src.dataset import PROMPT, PROMPT_ALLBOX_ALTFORM, PROMPT_ALTFORM, INSTRUCTION
from src.probing_utils import format_sentence, get_quantization_config, get_objects, get_box_ids, fix_random_seed, find_all


model_list = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    # local models
    "codellama/CodeLlama-13b-hf"
]

local_device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_REMOTE_ATTEMPTS = 5


# @nnsight.trace
def decode(pred, model):
    return model.tokenizer.decode(pred)
    # return model.tokenizer.convert_ids_to_tokens([pred], skip_special_tokens=True)[0].replace("▁", " ").replace("Ġ", " ") # generate strage tokens


# @nnsight.trace
def encode(input_example, model):
    # print(f"Encoding {input_example}")
    return model.tokenizer(input_example, return_tensors="pt")


def apply_encode(input_args):
    return input_args[0].tokenizer(input_args[1], return_tensors="pt")


def apply_decode(input_args):
    return input_args[0].tokenizer.decode(input_args[1], skip_special_tokens=True)


def setup_nnsight():
    """
    Setup script for nnsight
    """
    API = os.environ.get("NDIF_APIKEY")
    CONFIG.API.APIKEY = API
    assert "HF_TOKEN" in os.environ['']


def obj_name_to_index(obj_name, vocab_map, operation):
    if operation == "add":
        return vocab_map[obj_name] + 1
    elif operation == "remove":
        return vocab_map[obj_name]


def get_rowspace_projection(W: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/debias.py
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """
    if W.dim() == 1:
        W = W.unsqueeze(0)

    if torch.allclose(W, torch.zeros_like(W)):
        w_basis = torch.zeros_like(W.T)
    else:
        device = W.device
        w_basis = torch.Tensor(scipy.linalg.orth(W.cpu().T)).to(device)  # orthogonal basis doesn't deal with bf16
        # w_basis = torch.linalg.qr(W.T)[0]  # should be similar

    P_W = w_basis.matmul(w_basis.T)  # orthogonal projection on W's rowspace
    return P_W


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[torch.Tensor], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    # """
    # pdb.set_trace()
    I = torch.eye(input_dim)
    Q = torch.sum(torch.stack(rowspace_projection_matrices), dim=0)
    P = I - get_rowspace_projection(Q)

    return P


def debias_by_specific_directions(directions: List[torch.Tensor], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []
    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    if len(directions) == 1:
        P = rowspace_projections[0]
    else:
        P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    return P


def apply_projection(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    project the data X over the projection matrix P (to nullify some signal)
    """
    # return (P.dot(X.T)).T
    return X @ P.T.to(X.dtype).to(X.device)


def boost_or_negate_hidden(X: torch.Tensor, W: Union[List[torch.Tensor],torch.Tensor], direction: str, alpha: int, PNs: Optional[List[torch.Tensor]]) -> torch.Tensor:
    """
    boost or negate X in along the separation planes W
    
    Args:
        X (torch.Tensor, [bs, tokens, d_hidden]): the hidden state
        W (torch.Tensor, [n_weights, d_hidden]): the hidden state
    """
    if isinstance(W, list):
        W = torch.stack(W)
    if PNs is None:
        # Ps = [get_rowspace_projection(w) for w in W]  # I think we can batch this
        PNs = get_rowspace_projection(W)


    signs = torch.einsum("wh,bth->wbt", W.to(X.dtype), X).sign()  # [n_weights, bs, tokens]
    coef = torch.tensor(-1).pow(signs) if direction == "negate" else torch.tensor(-1).pow(1 - signs)  # [n_weights, bs, tokens]
    projected_ws = torch.stack([X - apply_projection(X, PN) for PN in PNs])  # [n_weights, bs, token, d_hidden]
    # sumed = torch.sum(coef * torch.Tensor(projected_ws), dim=0)
    sumed = torch.einsum("wbt,wbth->bth", coef, projected_ws)
    X += alpha * sumed 
    return X


def get_remove_phrases(op_phrases: List[str], query_op: bool, query_box_id: str) -> List[str]:
    """Filter from a list of operation phrases to get remove phrases (either for query or non-query boxes)"""
    remove_phrases = []
    for phrase in op_phrases:
        if "Remove " in phrase:
            if query_op and (query_box_id in phrase):
                remove_phrases.append(phrase)
            elif not query_op and (not query_box_id in phrase):
                remove_phrases.append(phrase)
        elif "Move " in phrase:
            if query_op and ((f"from Box {query_box_id}" in phrase) or (f"in Box {query_box_id}" in phrase)):
                remove_phrases.append(phrase)
            elif not query_op and (
            not ((f"from Box {query_box_id}" in phrase) or (f"in Box {query_box_id}" in phrase))):
                remove_phrases.append(phrase)
    return remove_phrases


def get_exist_phrases(phrases: List[str], query_op: bool, query_box_id: str, include_desc_phrases: bool = False) -> List[str]:
    """Filter from a list of operation phrases to get exist phrases (either for query or non-query boxes)"""
    exist_phrases = []
    for phrase in phrases:
        if ("contains " in phrase or " is in " in phrase) and include_desc_phrases:  # description phrase
            if query_op and (query_box_id in phrase):
                exist_phrases.append(phrase)
            elif not query_op and (not query_box_id in phrase):
                exist_phrases.append(phrase)
        elif "Put " in phrase:
            if query_op and (query_box_id in phrase):
                exist_phrases.append(phrase)
            elif not query_op and (not query_box_id in phrase):
                exist_phrases.append(phrase)
        elif "Move " in phrase:  # move into
            if query_op and (f" to Box {query_box_id}" in phrase):
                exist_phrases.append(phrase)
            elif not query_op and (not (f" to Box {query_box_id}" in phrase)):
                exist_phrases.append(phrase)
    return exist_phrases


def get_token_indices(sentence: str, phrase: str, box_id: str, obj: str, tokenizer: AutoTokenizer) -> Tuple[
    List[int], List[int]]:
    """
    Find all occurrence of box_id/obj after a phrase in token indices
    """
    # remove query phrase
    # pdb.set_trace(header="debugging get token indices")
    if "Statement" in sentence:  # few-shot
        sentence = sentence[:sentence.rfind("\nStatement:")]
    else:
        sentence = ". ".join(sentence.strip(".").split(". ")[:-1])

    box_id_token = tokenizer.encode(f" {box_id}", add_special_tokens=False)[-1]
    # most of time this is 1 token, but in llama tokenizer single digit is parsed to '', 'digit', so take last one
    # box_id_token = box_id_token[-1]
    obj_token = tokenizer.encode(f" {obj}", add_special_tokens=False)[-1]
    start_idx = len(tokenizer.encode(sentence[:sentence.rfind(phrase)]))
    tokens = tokenizer.encode(sentence)
    end_idx = len(tokens)
    box_id_indices = []
    obj_indices = []
    for i in range(start_idx, end_idx, 1):
        if tokens[i] == box_id_token:
            box_id_indices.append(i)
        elif tokens[i] == obj_token:
            obj_indices.append(i)

    return box_id_indices, obj_indices


def get_intervention_indices(dat: Dict[str, Any], formated_sentence: str, args, tokenizer, probe_type:str) -> Dict[Tuple[str, str], Dict[str, List[int]]]:
    """
    Given a datapoint, return dictionary of intervention phrase-item pair along with intervention token indices
    An example return object looks like:
    {
        ("Remove the ball from Box 3", "ball"): {
            "object": [10, 64],
            "box_id": [68]
        }
    }

    Args:
        dat: a dictionary datapoint
        formated_sentence: the formated sentence (w/ few-shot if included)
        args: experiment args

    """
    # NOTE the logic is broken when using few-shot prompt, maybe we can just truncate the formated sentence and only keep the queried part so that we can reuse the logic here. The format_sentence func seems ok so far.
    # there're two examples and one query. Each one starts with `Description:
    # pdb.set_trace(header="debugging get intervention indices")
    if "Description:" in formated_sentence:
        example_sent, query_sent = dat["prefix"].rsplit("Description: ", 1)
    else:
        example_sent = ""
        query_sent = dat["prefix"]
        
    # pdb.set_trace(header="debugging get intervention indices")
    desc_phrases = query_sent.split(". ")[0].split(", ")
    # op_phrases = query_sent.split()[1:-1] # not ". ", it ocul
    op_phrases =  re.split(r'\.\s|\.\n', query_sent)[1:-1]
    query_phrase = re.split(r'\.\s|\.\n', query_sent)[-1]
    query_box_id = query_phrase[-10] # in fewshot prompt there's a 'Statement: ' before the query phrase, so tracking backwards.
    # desc_phrases = dat["prefix"].split(". ")[0].split(", ")
    # op_phrases = dat["prefix"].split(". ")[1:-1]
    # query_phrase = dat["prefix"].split(". ")[-1]
    # query_box_id = query_phrase[4]

    # find all remove/exist phrase with object and box id
    if args.intervention_operation.endswith("remove"):
        phrases = get_remove_phrases(op_phrases, query_op=args.intervention_operation.startswith("query"), query_box_id=query_box_id)
    elif args.intervention_operation.endswith("exist"):
        phrases = get_exist_phrases(op_phrases, query_op=args.intervention_operation.startswith("query"), query_box_id=query_box_id, include_desc_phrases=False)
    elif args.intervention_operation.endswith("description"):
        phrases = get_exist_phrases(desc_phrases, query_op=args.intervention_operation.startswith("query"), query_box_id=query_box_id, include_desc_phrases=True)
    else:
        raise NotImplementedError(f"Intervention operation '{args.intervention_operation}' not implemented.")

    results = {}
    # find respective locations where they appear
    for phrase in phrases:
        # pdb.set_trace(header="debugging get token")
        objs = get_objects(phrase)
        # in these cases we only care about objects that end up in the query box, so remove ones that aren't relevant
        if args.intervention_operation.startswith("global"):
            objs = [o for o in objs if o in dat["gold_items"]]
        # TODO currently if global op move only the moved out box is considered
        box_id = query_box_id if "query" in args.intervention_operation else [i for i in get_box_ids(phrase) if query_box_id != i][0]

        for obj in objs:
            box_id_indices, obj_indices = get_token_indices(formated_sentence, phrase, box_id, obj, tokenizer)
            if args.intervention_site.endswith("op"):
                box_id_indices, obj_indices = [box_id_indices[0]], [obj_indices[0]]
            elif args.intervention_site.endswith("last"):
                box_id_indices, obj_indices = [box_id_indices[-1]], [obj_indices[-1]]
            results[(phrase, obj)] = defaultdict(list)
            if "number" in args.intervention_site or probe_type=="phrase":  # phrase probe need this to compute probe index, it gets deleted later
                results[(phrase, obj)]["box_id"] = box_id_indices
            if "object" in args.intervention_site or probe_type=="phrase":
                results[(phrase, obj)]["object"] = obj_indices
                
    # pdb.set_trace(header="debugging get intervention indices final")

    return results


def edit_hidden_given_probe_weights(
    hidden: torch.Tensor,
    object_weight: List[List[torch.Tensor]],
    box_id_weight: List[List[torch.Tensor]],
    patch_indices: Dict[str, List[int]],
    args
) -> torch.Tensor:
    """
    If we want to nullify, we project hiddens to null space of the probe weights

    obj_weight: List[List[torch.Tensor]], outter list corresponds to number of args.intervention_directions we want to do
        If we wnat to boost class 1 and null class 2, we can do that here. inner list corresponds to number of linearlly
        orthogonal probe weights from iterative Null space projection paper. usually is dim of 1 in this case.
    box_id_weights: similar as above
    """
    projected_hidden = []
    for site in [s for s in ["object","box_id"] if patch_indices[s]]:
        site_hidden = hidden[:, patch_indices[site], :]
        for i, intervention_direction in enumerate(args.intervention_direction):
            weight = locals()[f"{site}_weight"][i]
            projection = debias_by_specific_directions(weight, hidden.shape[-1])
            if intervention_direction == "null":
                site_hidden = apply_projection(site_hidden, projection)
            else:  # boost / negate
                site_hidden = boost_or_negate_hidden(X=site_hidden, W=weight,direction=intervention_direction, alpha=args.intervention_alpha,PNs=[projection])
        projected_hidden.append(site_hidden)

    projected_hidden = torch.cat(projected_hidden, 1) if len(projected_hidden) == 2 else projected_hidden[0]
    return projected_hidden

def prepare_weights_to_modify(
    object_weight: List[List[torch.Tensor]],
    box_id_weight: List[List[torch.Tensor]],
    patch_indices: Dict[str, List[int]],
    args,
    hs
) -> torch.Tensor:
    """
    Reorganizing the edit_hidden_given_probe_weights function to prepare weights only: we don't have access to layer
    output, so we're going to just prepare the projection matrices here and do projection inside nnsight trace
    Would it be faster if we move all the weights to cuda? Not sure if we actually use cuda here.
    """
    projections = []
    for site in [s for s in ["object","box_id"] if patch_indices[s]]:
        # pdb.set_trace(header="debugging prepare weights to modify")
        for i, intervention_direction in enumerate(args.intervention_direction):
            # check time usage
            st = time.time()
            weight = locals()[f"{site}_weight"][i]
            projection = debias_by_specific_directions(weight, hs) # hidden size of the model
            end = time.time()
            print(f"Time to compute projection for {site} direction {intervention_direction}: {end - st} seconds")
            print(f"Shape of projection: {projection.shape}; Size of projection: {projection.element_size() * projection.nelement() / 1e6} MB")
            if intervention_direction == "null":
                # site_hidden = apply_projection(site_hidden, projection)
                projections.append(projection)
            else:  # boost / negate
                # site_hidden = boost_or_negate_hidden(X=site_hidden, W=weight,direction=intervention_direction, alpha=args.intervention_alpha,PNs=[projection])
                raise NotImplementedError("boost/negate not implemented in prepare weights function yet.")
        # projected_hidden.append(site_hidden) can be done inside tracing

    # projected_hidden = torch.cat(projected_hidden, 1) if len(projected_hidden) == 2 else projected_hidden[0] can be done inside tracing
    return projections  

def get_intervention_success(target_item: str, intervened_predicted_items: List[str], args):
    """
    The assumption here is the script caller knows what they are doing
    if we are trying to reverse "remove tag", then we want to see that obj predicted, 
    and this can be either through boosting the exist class direction, or null/negate 
    the remove class direction.
    """
    if args.intervention_operation in ["query-remove", "global-remove"]:
        # if args.intervention_direction in ["null", "negate"]:
        return target_item in intervened_predicted_items
        # else:  # boost
        #     return target_item not in intervened_predicted_items
    if args.intervention_operation in ["query-exist", "query-description"]:
        # if args.intervention_direction in ["null", "negate"]:
        return target_item not in intervened_predicted_items
        # else:
        #     return target_item in intervened_predicted_items
    raise NotImplementedError()


def get_intervention_rest_success(target_item: str, orig_items: List[str], intervened_items: List[str]):
    """ Check if the rest of the items are predicted as before / unaffected by intervention """
    o_items = set([o for o in orig_items if o != target_item])
    i_items = set([i for i in intervened_items if i != target_item])
    return o_items == i_items


def get_probe_weights(args, d_hidden, probes_layer, probe_token: str, probe_type: str):
    if probe_token == "object" and probe_type == "span":
        weights = [p['proj.weight'][1, d_hidden:] for p in probes_layer]
    elif probe_token == "number" and probe_type == "span":
        weights = [p['proj.weight'][1, :d_hidden] for p in probes_layer]
    elif probe_type == "phrase":
        # this one makes more sense, and empirically is better than the one commented out
        weights = [[p['proj.weight'].reshape(700, 3, -1)[:, probe_class] for p in probes_layer] for probe_class in args.intervention_probe_class]
    else:
        raise NotImplementedError

    return weights


def get_op_indices(sentence:str) -> List[int]:
    if "Statement:" in sentence: # remove fs prompts
        sentence = sentence[sentence.rfind("Description: "):].repace("\nStatement: ", "").strip()
    query_box = int(sentence.split()[-2])
    sent_no_query = sentence[:sentence.rfind(".")]

    op_indices = []
    for box_id_idx in find_all(sent_no_query, str(query_box)):
        sent_before = sent_no_query[:box_id_idx]
        op_idx = sent_before.count(",") + sent_before.count(".")
        op_indices.append(op_idx)
    return op_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_dir", type=str, required=True,
                        help="(List of) Path(s) to probe checkpoints. If list, assume it's result of iterative nullified probe weights",
                        default="/projectnb/tin-lab/sebastian/box-models/probe_checkpoints/codellama-13b/probing/state_binary_exclude_empty")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Path to the entity tracking model.",
        default="projectnb/tin-lab/sebastian/box-models/checkpoints/CodeLlama-13b-hf"
        # TODO: In practice should be a hosted model
    )
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model with 8-bit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model with 4-bit")

    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/objects_with_bnc_frequency.csv",
        help="Path to the entity vocabulary .csv file."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Path to a .jsonl file containing the output from compute_metrics.",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/intervention"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="If remote, code is running on NDIF, otherwise local small model"
    )
    parser.add_argument("--layers", type=int, default=None, help="layer to intervene (start, at, or until).")
    parser.add_argument("--intervention_layer_type", type=str, default="last_n",
                        help="Intervene on last_n, first_n, or just at n-th layer.",
                        choices=["last_n", "first_n", "at_n"])
    parser.add_argument("--scaling_factor", type=int, default=1.0, help="A constant to scale the intervention vector by.")
    parser.add_argument("--sampling_seed", type=int, default=22, help="Seed for random sampling.")
    parser.add_argument(
        "--intervention_operation", type=str, default="query-remove",
        choices=["query-remove", "query-exist", "query-description", "global-remove", "global-exist", "global-description"],
    )
    parser.add_argument(
        "--intervention_probe_class", type=str, default='2',
        help="Phrase probe only, (comma separated list of) class idx of the ternary probe weight we want to use",
        # choices=[0, 1, 2],  # 0: nonexist, 1: exist, 2: removed
    )
    parser.add_argument(
        "--intervention_direction", type=str, default="null",
        #choices=["boost", "negate", "null"],
        help='Comma separated list of direction of intervention, ["boost", "negate", "null"]',
    )
    parser.add_argument(
        "--intervention_site", type=str, default="first",
        choices=[
            # "first", "all",
            "last",
            "number-object-op", "number-object-all", "number-object-last",  # number = box_id
            "number-op", "number-all", "number-last",
            "object-op", "object-all", "object-last",
        ],
        help="Intervention index. 'last'=last token 'the', with box_id projection."
             "'op'=intervene at operation phrase only, 'last'=intervene at last occurrence of the obj/box_id, "
             "'all'=intervene at all occurrences of the obj/box_id, after the operation phrase"
    )
    parser.add_argument(
        "--intervention_alpha", type=float, default=1,
        help="the strength for 'boost' or 'negate' intervention."
    )

    parser.add_argument(
        "--few_shot_prompt",
        action="store_true"
    )
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Add random weights (for debugging)."
    )
    parser.add_argument(
        "--normalization",
        action="store_true",
        help="Add normalization to the intervention vector."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="how many tokens to generate."
    )


    parser.add_argument(
        "--cache_projections",
        action="store_true",
        default=False,
        help="Whether to cache each layer's projections as h5 file.",
    )


    # argument on how to sample the datapoints to intervene on
    parser.add_argument(
        "--eval_local_numops", type=str, default="1",
        help="Comma separated list of number of local operations to sample"
    )
    parser.add_argument(
        "--eval_local_op_index", type=str, default="",
        help="Comma separated list of number of local operations index to sample. If you want to sample query operation "
             "being the 1st operation, put 7 (because index 0-6 are for description phrase). default none"
    )
    parser.add_argument(
        "--eval_sample_per_numops", type=int, default=100,
        help="number of samples to evaluate per num-ops"
    )
    parser.add_argument(
        "--eval_num_gold_items", type=int, default=None,
        help="filter for samples with specific number of gold items"
    )
    parser.add_argument(
        "--filter_correct", type=str, default="1",
        help="Intervene on 1 (success only) or 0 (failed only), or None (not filtering)"
    )
    parser.add_argument(
        "--prompt_format", type=str, default="default",
        help="Prompt format to use"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=1,
    )
    

    args = parser.parse_args()
    args.probe_dir = args.probe_dir.split(",")
    args.eval_local_numops = [int(i) for i in args.eval_local_numops.split(",")]
    args.eval_local_op_index = [int(i) for i in args.eval_local_op_index.split(",")] if args.eval_local_op_index else []
    args.intervention_probe_class = [int(i) for i in args.intervention_probe_class.split(",")]
    args.intervention_direction = args.intervention_direction.split(",")

    fix_random_seed(args.sampling_seed)

    if args.remote:
        setup_nnsight()


    q_config = get_quantization_config(args)
    model = LanguageModel(args.model_name, device_map="auto", quantization_config=q_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    n_layers = model.config.num_hidden_layers
    d_hidden = model.config.hidden_size
    probe_type = "span" if "span" in args.probe_dir else "phrase"

    print(f'Model loaded:{args.model_name}, layers: {n_layers}, hidden dims: {d_hidden}, {args.intervention_layer_type}={args.layers}')

    object_map = {}
    object_list = []
    with open(args.vocab_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            object_map[row["\ufeffobject_name"]] = i
            object_list.append(row["\ufeffobject_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # probes are list (layers) of list (iterative probes)
    probes = [[] for _ in range(n_layers)]
    intervention_layers = [args.layers-1] if args.intervention_layer_type =="at_n" else range(args.layers) if args.intervention_layer_type=="first_n" else range(n_layers-args.layers, n_layers)

    for layer_n in intervention_layers:
        for probe_dir in args.probe_dir:
            probes[layer_n].append(torch.load(os.path.join(probe_dir, f"layer{layer_n + 1}_token1/checkpoint.ckpt"), map_location=torch.device("cpu")))
            probes[layer_n][-1].requires_grad = False

    if args.intervention_layer_type in ["first_n", "last_n"]:
        scaling_factor = args.scaling_factor / args.layers

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"intervention_{os.path.basename(os.path.normpath(args.model_name))}"
        f"{'_8bit' if args.load_in_8bit else '_4bit' if args.load_in_4bit else ''}"
        f"_{args.intervention_operation}_{args.intervention_direction[0] if len(args.intervention_direction)==1 else args.intervention_direction}_{args.intervention_site}"
        f"_alpha={args.intervention_alpha}_"
        f"{'ln=' if args.intervention_layer_type =='last_n' else 'fn=' if args.intervention_layer_type =='first_n' else 'n='}{args.layers}.jsonl",
    )
    
    df = pd.read_json(args.results_path, lines=True, orient="records")
    df = df[df["correct"] == int(args.filter_correct)] if args.filter_correct is not None else df
    df["query_op_indices"] = df.prefix.apply(lambda s: get_op_indices(s))
    sampled_dfs = []

    for i in args.eval_local_numops:
        sample_df = df[(df["numops"] == i)]  # local op
        sample_df = sample_df[sample_df["gold_answer"] != "nothing"]
        if args.eval_num_gold_items is not None:
            sample_df = sample_df[sample_df["gold_items"].apply(lambda g: len(g)==args.eval_num_gold_items)]
        if args.eval_local_op_index:
            sample_df = sample_df[sample_df["query_op_indices"].apply(lambda indices: any(i in args.eval_local_op_index for i in indices))]
        sampled_dfs.append(sample_df.sample(n=min(args.eval_sample_per_numops, len(sample_df)), random_state=args.sampling_seed))

    sampled_dfs = pd.concat(sampled_dfs)
    sampled_data = sampled_dfs.to_dict("records")
    print(f"Number of data sampled: {len(sampled_data)}. (note, not all datapoints have desired query operations)")
    print(sampled_dfs["numops"].value_counts())
    if os.path.exists(out_path):
        print(f"found previous results, removing {out_path}")
        os.remove(out_path)

    num_generated = 0
    
    obj_weights = []
    box_id_weights = []
    
    if args.intervention_layer_type in ["first_n", "at_n"]:
        layers = range(args.layers) if args.intervention_layer_type=="first_n" else [args.layers-1]
        for n in tqdm(layers, desc="Computing projections"):
            if args.random_weights:
                obj_probe_vectors.append(scaling_factor * torch.randn((1, d_hidden), dtype=torch.bfloat16))
                box_id_probe_vectors.append(scaling_factor * torch.randn((1, d_hidden), dtype=torch.bfloat16))
            else:
                # right now just take class 1 (i.e. remove/exist) weights, could also do difference
                if probe_type == "span" or "object" in args.probe_dir[0]:  # span probe weights are 2 X h_dim
                    obj_weight = get_probe_weights(args, d_hidden, probes[n], probe_token="object", probe_type=probe_type)
                else:
                    obj_weight = []
                obj_weights.append(obj_weight)
                if probe_type == "span" or "number" in args.probe_dir[0]:
                    box_id_weight = get_probe_weights(args, d_hidden, probes[n], probe_token="number", probe_type=probe_type)
                else:
                    box_id_weight = []
                box_id_weights.append(box_id_weight)

        
    elif args.intervention_layer_type == "last_n":
        layers = range(args.layers, 0, -1)  # old behavior range(args.last_n, 0, -1)
   
        for n in tqdm(layers, desc="Computing projections"):
            if args.random_weights:
                obj_probe_vectors.append(scaling_factor * torch.randn((1, d_hidden), dtype=torch.bfloat16))
                box_id_probe_vectors.append(scaling_factor * torch.randn((1, d_hidden), dtype=torch.bfloat16))
            else:
                # right now just take class 1 (i.e. remove/exist) weights, could also do difference
                if probe_type == "span" or "object" in args.probe_dir[0]:  # span probe weights are 2 X h_dim
                    obj_weight = get_probe_weights(args, d_hidden, probes[-n], probe_token="object", probe_type=probe_type)
                else:
                    obj_weight = []
                obj_weights.append(obj_weight)
                if probe_type == "span" or "number" in args.probe_dir[0]:
                    box_id_weight = get_probe_weights(args, d_hidden, probes[-n], probe_token="number", probe_type=probe_type)
                else:
                    box_id_weight = []
                box_id_weights.append(box_id_weight)

    
    for idx in tqdm(range(0, len(sampled_data), args.batch_size), desc="Intervening over data"):
        actual_bs = min(args.batch_size, len(sampled_data)-idx)
        batch_data = sampled_data[idx: idx + actual_bs]
        batch_input_examples = [] # note: this list should be of length actual_bs * num_phrases_per_example, not actual_bs, cuz we need to unroll the inner loop
        
        for dat in batch_data:
            
            example_sent = format_sentence(dat, args.prompt_format, None, model_name=args.model_name)
            target_box_num = int(example_sent.split()[-2])
            target = dat["gold_answer"]  # ground truth
            example_sent += " the"
            orig_items = [] if target == "nothing" else target.removeprefix("the ").split(" and the ")
            # get a list of indices, where each item in list represent a set of intervention for a particular
            # removed or existing objects, and their occurrence in the sentence
            intervention_indices = get_intervention_indices(dat, example_sent, args, tokenizer, probe_type)
            print(f"datapoint exploded to {len(intervention_indices)} interventions (box_id-object pairs)")
            
            for patch_phrase, patch_indices in intervention_indices.items():
                input_example = tokenizer(example_sent, return_tensors="pt")
                num_generated += 1

                # if phrase probe, find the object, and box_id to get the right probe weights
                if probe_type == "phrase":
                    # pdb.set_trace()
                    obj_str = tokenizer.decode(input_example["input_ids"][0, patch_indices["object"]], skip_special_tokens=True).strip()
                    box_id = int(tokenizer.decode(input_example["input_ids"][0, patch_indices["box_id"]], skip_special_tokens=True).strip())
                    phrase_probe_indices = box_id * 100 + object_map[obj_str]
                    if "object" in args.probe_dir[0]:
                        # there's logic for object probe patching, need to ask if this should be run too.
                        del patch_indices["box_id"]
                    else:
                        del patch_indices["object"]
                
                    intervention_layer_type = args.intervention_layer_type
                    layers = args.layers    
                    
                    model_layers = enumerate(model.model.layers[:layers]) if intervention_layer_type=="first_n" else enumerate([model.model.layers[layers-1]])
                    projections_ls = []
                    if intervention_layer_type in ["first_n", "at_n"]:  # might be more efficient than patch all layers
                        for idx, layer in model_layers:
                            all_patch_indices = [*patch_indices["box_id"], *patch_indices["object"]]
                            object_weight = obj_weights[idx] if (probe_type == "span") or (not obj_weights[idx]) else [[w[phrase_probe_indices] for w in probe_class_weights] for probe_class_weights in obj_weights[idx]]
                            box_id_weight = box_id_weights[idx] if (probe_type == "span") or (not box_id_weights[idx]) else [[w[phrase_probe_indices] for w in probe_class_weights] for probe_class_weights in box_id_weights[idx]]
                            projections = prepare_weights_to_modify(
                                object_weight=object_weight,
                                box_id_weight=box_id_weight,
                                patch_indices=patch_indices,
                                args=args,
                                hs=d_hidden
                            )
                            projections_ls.append(projections) # its a list still
                    elif intervention_layer_type=="last_n":  # might be more efficient than patch all layers
                        for idx, layer in enumerate(model.model.layers[-layers:]):
                            # NOTE: This is more efficient, seems not that easy to get stuck in NDIF
                            # layer.output[0][:, patch_indices, :] += obj_probe_vectors[idx-args.last_n]
                            all_patch_indices = [*patch_indices["box_id"], *patch_indices["object"]]
                            object_weight = obj_weights[idx-layers] if (probe_type == "span") or (not obj_weights[idx-layers]) else [[w[phrase_probe_indices] for w in probe_class_weights] for probe_class_weights in obj_weights[idx-layers]]
                            box_id_weight = box_id_weights[idx-layers] if (probe_type == "span") or (not box_id_weights[idx-layers]) else [[w[phrase_probe_indices] for w in probe_class_weights] for probe_class_weights in box_id_weights[idx-    layers]]
                            # prepare projections here
                            projections = prepare_weights_to_modify(
                                object_weight=object_weight,
                                box_id_weight=box_id_weight,
                                patch_indices=patch_indices,
                                args=args,
                                hs=d_hidden
                            )
                            projections_ls.append(projections)
                batch_input_examples.append({
                    "input_ids": input_example["input_ids"],
                    "attention_mask": input_example["attention_mask"],
                    "patch_indices": patch_indices,
                    "projections": projections_ls,
                    "patch_phrase": patch_phrase,
                    "example_sent": example_sent,
                    "orig_items": orig_items,
                    "target_box_num": target_box_num,
                    "dat": dat,
                    "all_patch_indices": all_patch_indices,
                })
                
            # now we have prepared all inputs for this batch
            # Right here we group them into a single batch for generation. What we're trying to do here, is, convert dict to batched tensors, add padding and change indexing correspondingly.
        tokenizer.pad_token_id = tokenizer.eos_token_id  # set pad token id
        input_ids_list = [be["input_ids"][0] for be in batch_input_examples]
        attention_mask_list = [be["attention_mask"][0] for be in batch_input_examples]
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        batched_projections = torch.stack(
            [torch.stack(
                [torch.stack(mat) for mat in be['projections']]
            ) for be in batch_input_examples]
        ).to(torch.bfloat16)  # use float16 to save memory, shape [bs, num_layers, num_projections_mat_this_layer, d, d]
        # reduce the num_projections_mat_this_layer dimension by merging them into a single projection matrix. it's actually always 1
        batched_projections = batched_projections.squeeze(2)  # now shape [bs, num_layers, d, d]
        
        # now each projection is [[mat1], [mat2], ...], need to stack them accordingly
        # projections is a little bit tricky, its a list of [d,d] matrics, and we need to format it 
        # batched_projections = torch.stack([torch.stack(be["projections"]) for be in batch_input_examples]).to(torch.bfloat16).squeeze() # use float16 to save memory
        batched_all_patch_indices = [be["all_patch_indices"] for be in batch_input_examples]
        # to tensor
        # we need to also adjust the patch indices accordingly
        input_id_lens = [ids.shape[0] for ids in input_ids_list]
        adjusted_batch_patch_indices = {}
        for site in ["object", "box_id"]:
            adjusted_batch_patch_indices[site] = []
        for i, be in enumerate(batch_input_examples):
            pad_len = batch_input_ids.shape[1] - input_id_lens[i]
            adjusted_patch_indices = {}
            for site in be["patch_indices"]:
                adjusted_patch_indices[site] = [idx + pad_len for idx in be["patch_indices"][site]]
                adjusted_batch_patch_indices[site].append(adjusted_patch_indices[site])

        
        for n_attempt in range(MAX_REMOTE_ATTEMPTS):
            try:
                with torch.no_grad():
                    with model.generate({"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}, max_new_tokens=args.max_new_tokens, remote=args.remote):
                        # patch first n layers
                        # only patch the last arg.last_n layers DEBUG
                        if intervention_layer_type in ["first_n", "at_n"]:  # might be more efficient than patch all layers
                            model_layers = enumerate(model.model.layers[:layers]) if intervention_layer_type=="first_n" else enumerate([model.model.layers[layers-1]])
                            for idx, layer in model_layers:
                                projection = batched_projections[:, idx] # now this should be a tensor list
                                projected_hidden = []
                                for site in [s for s in ["object","box_id"] if adjusted_batch_patch_indices[s] and len(adjusted_batch_patch_indices[s][0])>0]:
                                    #NOTE added length > 0 logic to avoid empty indexing error
                                    hidden = layer.output # now that layer.output returns the tensor, instead of the tuple, so we dont need to do layer.output[0]
                                    # site_hidden = hidden[:, adjusted_batch_patch_indices[site]]
                                    # print(f"hidden shape{hidden.shape}, adjusted indices shape {torch.tensor(adjusted_batch_patch_indices['box_id']).shape}")
                                    site_hidden = hidden[torch.arange(hidden.shape[0]), torch.tensor(adjusted_batch_patch_indices[site]).squeeze(-1).to(int).to(hidden.device)] # bs, num_patches?, dh
                                    # X @ P.T.to(X.dtype).to(X.device)
                                    # print(f"Applying projection at layer {idx} for site {site} with shape {site_hidden.shape} and projection shape {projection.shape}, hidden shape {hidden.shape}")
                                    # site_hidden = site_hidden @ projection.T.to(site_hidden.dtype).to(site_hidden.device)
                                    site_hidden = torch.bmm(projection.to(site_hidden.dtype).to(site_hidden.device), site_hidden.unsqueeze(-1)).squeeze(-1)  # bs, num_patches, dh
                                    projected_hidden.append(site_hidden)
                                projected_hidden = torch.cat(projected_hidden, 1) if len(projected_hidden) == 2 else projected_hidden[0]                        
                                layer.output[torch.arange(layer.output.shape[0]), torch.tensor(batched_all_patch_indices).squeeze(-1).to(int)] = projected_hidden # this need to be fixed to accomodate variable batch patch indices
                        # only patch the last arg.last_n layers
                        if intervention_layer_type=="last_n":  # might be more efficient than patch all layers
                            for idx, layer in enumerate(model.model.layers[-layers:]):
                                # NOTE: This is more efficient, seems not that easy to get stuck in NDIF
                                # layer.output[0][:, patch_indices, :] += obj_probe_vectors[idx-args.last_n]
                                projection = batched_projections[:, idx - layers]  # now this should be a tensor list
                                projected_hidden = []
                                for site in [s for s in ["object","box_id"] if patch_indices[s] and len(adjusted_batch_patch_indices[s][0])>0]:
                                    hidden = layer.output
                                    # site_hidden = hidden[:, adjusted_batch_patch_indices[site]] # bs, num_patches?, dh
                                    # X @ P.T.to(X.dtype).to(X.device)
                                    site_hidden = hidden[torch.arange(hidden.shape[0]), torch.tensor(adjusted_batch_patch_indices[site]).squeeze(-1).to(int)]
                                    print(site_hidden.shape, projection.shape)
                                    # site_hidden = site_hidden @ projection.T.to(site_hidden.dtype).to(site_hidden.device)
                                    site_hidden = torch.bmm(projection.to(site_hidden.dtype).to(site_hidden.device), site_hidden.unsqueeze(-1)).squeeze(-1)  # bs, num_patches, dh
                                    projected_hidden.append(site_hidden)    
                                projected_hidden = torch.cat(projected_hidden, 1) if len(projected_hidden) == 2 else projected_hidden[0]                    
                                layer.output[torch.arange(layer.output.shape[0]), torch.tensor(adjusted_batch_patch_indices).squeeze(-1).to(int)] = projected_hidden
                        output = model.generator.output.save()
                break
            except Exception as e:
                print(f"Remote generation attempt {n_attempt+1} failed with error: {e}")
                if n_attempt == MAX_REMOTE_ATTEMPTS - 1:
                    raise e
                else:
                    print("Retrying...")
        
        full_generation = model.tokenizer.batch_decode(output[:,-args.max_new_tokens:], skip_special_tokens=True)
        
        # # TODO: NNsight cannot do early stopping in the middle of a session, which is suprising(see: https://github.com/ndif-team/nnsight/issues/399). So for now we just do a fixed number of generations, and remove everything after the first period. I think 15 new tokens should be fine for starter.
        generation_ls = [f.split(".")[0].split("Box")[0].strip() + "." for f in full_generation]

        for gen_idx, generation in enumerate(generation_ls):
            raw_generation = full_generation[gen_idx]
            patch_phrase = batch_input_examples[gen_idx]["patch_phrase"]
            example_sent = batch_input_examples[gen_idx]["example_sent"]
            orig_items = batch_input_examples[gen_idx]["orig_items"]
            target_box_num = batch_input_examples[gen_idx]["target_box_num"]
            dat = batch_input_examples[gen_idx]["dat"]
            intervened_items = list(set([o for o in generation.lower().replace(",", " ").replace(".", " ").split(" ") if o in object_map]))
            
            if "Description" in example_sent:
                _, query_sent = dat["prefix"].rsplit("Description: ", 1)
                og_query_box_objs = get_objects(query_sent.split("Description: ")[0].split(",")[int(target_box_num)])
            else:
                og_query_box_objs = get_objects(example_sent.split(".")[0].split(",")[int(target_box_num)])
            # whether specific intervened item is changed
            intervention_obj_success = get_intervention_success(patch_phrase[1], intervened_items, args)
            intervention_rest_success = get_intervention_rest_success(patch_phrase[1], orig_items, intervened_items)
            before_target_phrase = example_sent[:example_sent.find(patch_phrase[0])]
            intervention_target_phrase_index = before_target_phrase.count(",")+before_target_phrase.count(".")
            write_d = {
                'prefix': example_sent,
                'original_answer': dat['original_answer'],
                'parsed_original_answer': dat['parsed_original_answer'],
                'gold_answer': orig_items,
                'intervened_answer': full_generation,
                'intervened_answer_items': intervened_items,
                'intervention_target_phrase': patch_phrase[0],
                'intervention_target_phrase_index': intervention_target_phrase_index,
                'intervention_target_item': patch_phrase[1],
                'intervention_target_obj_idx_in_phrase': og_query_box_objs.index(patch_phrase[1]) if patch_phrase[1] in og_query_box_objs else None,
                'intervention_operation': args.intervention_operation,
                'intervention_direction': args.intervention_direction[0] if len(args.intervention_direction)==1 else args.intervention_direction,
                'intervention_site': args.intervention_site,
                'intervention_obj_success': intervention_obj_success,
                'intervention_rest_success': intervention_rest_success,
                'intervention_probe_class': args.intervention_probe_class[0] if len(args.intervention_probe_class)==1 else args.intervention_probe_class,
                'intervention_alpha': args.intervention_alpha,
                'numops': dat["numops"],
                'numops_global': dat["numops_global"],
                'gold_num_items': len(orig_items),
                'target_box_num': target_box_num,
                'scaling_factor': args.scaling_factor,
                'intervention_layer_type': args.intervention_layer_type,
                'layers':args.layers,
                'raw_generation': raw_generation
            }

            with open(out_path, 'a') as wf:
                wf.write(json.dumps(write_d) + "\n")

    


if __name__ == '__main__':
    main()