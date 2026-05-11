import argparse

from tqdm import tqdm
import numpy as np
import torch

import sys
sys.path.append("..")
from utils import get_model_and_tokenizer, get_random_guess_baseline, fix_random_seed, get_random_circuit, get_circuit, get_mean_activations, eval_circuit_performance, get_root_exp_dir, MODEL_TO_SHORT, setup_nnsight
from patch_utils import build_parser, post_arg_parse_fix, get_model_and_dataset


def eval_model_performance(model, dataloader):
    """
    Evaluate first token prediction correctness
    TODO extend to multi obj?
    """
    total_count = 0
    argmax_correct_any = 0
    argmax_correct_full = []
    topk_correct_full = []

    with torch.no_grad():
        for output in tqdm(dataloader):
            for k, v in output.items():
                if v is not None and isinstance(v, torch.Tensor):
                    output[k] = v.to(model.device)

            logits = model(input_ids=output["base_tokens"]).logits
            for bi in range(len(output["labels"])):
                labels = output["labels"][bi]  # multiple target objects
                topk_pred = torch.argsort(logits[bi][output["base_last_token_indices"][bi]], descending=True)[:len(labels)].cpu().numpy()
                if (topk_pred[0] == labels).sum() > 0:
                    argmax_correct_any += 1

                argmax_correct_full_batch = []
                topk_correct_full_batch = []
                for k, label in enumerate(labels):
                    argmax_correct_full_batch.append(1 if topk_pred[0] == label > 0 else 0)
                    topk_correct_full_batch.append(1 if (topk_pred == label).sum() > 0 else 0)

                total_count += 1
                argmax_correct_full.append(argmax_correct_full_batch)
                topk_correct_full.append(topk_correct_full_batch)

    del logits
    torch.cuda.empty_cache()
    current_acc = round(argmax_correct_any / total_count, 2)
    return current_acc, argmax_correct_full, topk_correct_full


def eval_circuit_main(args: argparse.Namespace):
    """
    evaluate model performance, circuit performance and a random circuit (same size)
    performance
    """
    if args.remote:
        setup_nnsight()

    dataloader, dataset, model = get_model_and_dataset(args)
    #circuit_components, _, _, _ = get_circuit(model, args.circuit_root_path, args.n_groupA, args.n_groupB, args.n_groupC, args.n_groupD, top_p=args.top_p)

    # Gabriel: After using the knee to find the circuits, n_group variables are not used
    circuit_components, groupA, groupB, groupC, groupD = get_circuit(model, args.circuit_root_path, n_more_heads_per_group=args.n_more_heads_per_group)

    n_groupA = len(groupA)
    n_groupB = len(groupB)
    n_groupC = len(groupC)
    n_groupD = len(groupD)

    print(f"len(groupA)={len(groupA)}")
    print(f"len(groupB)={len(groupB)}")
    print(f"len(groupC)={len(groupC)}")
    print(f"len(groupD)={len(groupD)}")


    model_acc, model_argmax_full, model_topk_full = eval_model_performance(model, dataloader)
    if np.array([len(p) for p in model_argmax_full]).std() == 0:
        model_argmax_full = np.array(model_argmax_full).sum(0)/len(model_argmax_full)
        model_topk_full = np.array(model_topk_full).sum(0) /len(model_topk_full)
        print(f"Model Performance {model_acc}. Argmax accuracy by label index: {model_argmax_full}. TopK accuracy by label index: {model_topk_full} \n")
    else:
        print(f"Model Performance {model_acc}\n")

    # mean activation data also needs to be loaded filtered by operation orders
    mean_activations, modules = get_mean_activations(model=model, args=args, cache_dir=args.mean_activation_cache_path)

    circuit_acc, c_argmax_full, c_topk_full = eval_circuit_performance(
        model, dataloader, modules, circuit_components, mean_activations, ablate_non_vital_pos=not args.skip_ablate_non_vital_pos,
    )
    if np.array([len(p) for p in c_argmax_full]).std() == 0:
        c_argmax_full = np.array(c_argmax_full).sum(0)/len(c_argmax_full)
        c_topk_full = np.array(c_topk_full).sum(0) /len(c_topk_full)
        print(f"Circuit Performance {circuit_acc}. Argmax accuracy by label index: {c_argmax_full}. TopK accuracy by label index: {c_topk_full} \n")
    else:
        print(f"Circuit Performance {circuit_acc}\n")


    random_circuit_acc = 0
    random_circuit_argmax_full = []
    random_circuit_topk_full = []
    n_iters = 10
    for i in range(n_iters):
        random_circuit_components = get_random_circuit(model, n_groupA, n_groupB, n_groupC, n_groupD)
        r_circuit_acc, rc_argmax_full, rc_topk_full = eval_circuit_performance(
            model, dataloader, modules, random_circuit_components, mean_activations
        )
        random_circuit_acc += r_circuit_acc
        if np.array([len(p) for p in rc_argmax_full]).std() == 0:
            rc_argmax_full = np.array(rc_argmax_full).sum(0) / len(rc_argmax_full)
            rc_topk_full = np.array(rc_topk_full).sum(0) / len(rc_topk_full)
            print(f"Random Circuit {i} Performance {r_circuit_acc}. Argmax accuracy by label index: {rc_argmax_full}. TopK accuracy by label index: {rc_topk_full} \n")
            random_circuit_argmax_full.append(rc_argmax_full)
            random_circuit_topk_full.append(rc_topk_full)
        else:
            print(f"Random Circuit {i} Performance {r_circuit_acc}\n")
    random_circuit_acc = round(random_circuit_acc / n_iters, 2)
    print(f"Random Circuit Average Performance {random_circuit_acc}")
    if len(random_circuit_argmax_full) > 0:
        print(f"Argmax accuracy by label index: {np.array(random_circuit_argmax_full).mean(0)}")
        print(f"Topk accuracy by label index: {np.array(random_circuit_topk_full).mean(0)}")

    print(f"Faithfulness (Circuit): {round(circuit_acc / model_acc, 2)}")
    print(f"Faithfulness (Random Circuit): {round(random_circuit_acc / model_acc, 2)}")
    return


def add_args(parser: argparse.ArgumentParser):
    """
    circuit_root_path: str = "../outputs/nnsight_patch_no_op/gemma-2-2b/n200",
    percentage: float = 0.3,
    minimality_threshold: float = 0.01,
    """
    parser.add_argument('--circuit_root_path', help='where circuit info dir lives', type=str, default="../outputs/nnsight_patch_noop/gemma-2-2b/n200")
    parser.add_argument('--skip_ablate_non_vital_pos', help='skip ablation on non-essential tokens', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mean_activation_cache_path', help='path to cache mean activations', type=str, default="../outputs/nnsight_patch_noop/gemma-2-2b")
    parser.add_argument("--n_more_heads_per_group", help="number of additional heads per group from knee methods", type=int, default=0)
    return parser

if __name__ == "__main__":
    parser = add_args(build_parser())
    args = parser.parse_args()
    print(f"ARGS: {args}")
    post_arg_parse_fix(args)
    fix_random_seed(args.seed)
    eval_circuit_main(args)
