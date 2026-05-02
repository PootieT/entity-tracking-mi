import argparse
import json
import random
import re
import sys

_MODIFIERS = ["big", "small", "blue", "green", "red", "yellow"]
_MODIFIERS_REGEX_STR = "(" + "|".join(_MODIFIERS) + ")"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--max_numops",
        type=int,
        default=7
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100
    )
    
    parser.add_argument(
        "--include_entire_state",
        action="store_true"
    )
    
    parser.add_argument(
        "--pragmatics",
        action="store_true"
    )
    
    parser.add_argument(
        "--move_contents",
        action="store_true"
    )

    
    
    return parser.parse_args()

def is_pragmatic(d):
    pragmatic = False
    s = d["sentence"]
    op_start = s[:-1].find(".")
    op_end = s[:-1].rfind(".")
    if op_end != op_start:
        op_string = s[op_start:op_end]
        if "contains nothing" in s[op_end:-1] or "is empty" in s[op_end:-1]:
            return pragmatic 
        box_contents_string = s[op_end:-1].split(" contains ")[1].strip()
        gold_items = set([i.replace("the ", "") for i in box_contents_string.split(" and ")])
        for gold_item in gold_items:
            if " " in gold_item:
                gold_item_type = gold_item.split(" ")[1]
                if f"the {gold_item_type}" in op_string:
                    uses = re.findall(f"{_MODIFIERS_REGEX_STR} {gold_item_type}", s)
                    pragmatic = pragmatic or len(set(uses)) > 1
    return pragmatic


def has_move_content(d):
    involves_move_content = False
    s = d["sentence"]
    box_start = s[:-1].rfind("Box ") + 4
    box_no = s[box_start:box_start+1]
    move_ops1 = re.findall(f"Move the contents of Box [0-9] to Box {box_no}", s)
    involves_move_content = len(move_ops1) > 0 

    return involves_move_content

    

if __name__ == "__main__":
    args = parse_args()
    
    if (args.pragmatics or args.move_contents) and not args.include_entire_state:
        raise argparse.ArgumentError(None, "--pragmatics and --move_contents have to be used together with --include_entire_state")
    
    data_points = {}
    with open(args.input_file, encoding="utf-8") as in_f:
        for line in in_f:
            d = json.loads(line)
            numops = d["numops"]
            if numops > args.max_numops:
                continue
        
            if args.pragmatics:
                d["is_pragmatic"] = is_pragmatic(d)
                
            if args.move_contents:
                d["has_move_content"] = has_move_content(d)
        
            if numops not in data_points:
                data_points[numops] = []
            data_points[numops].append(d)
            
    
    if not args.include_entire_state:
        for numops, exs in data_points.items():
            sample = None
            if len(exs) < args.max_samples:
                sample = exs
            else:
                sample = random.sample(exs, args.max_samples)
            
            for ex in sample:
                print(json.dumps(ex))
    else:
        sample_ids = {}
        for numops, exs in data_points.items():
            sample = None
            
            # Sample primarily examples that require pragmatics to solve."
            if args.pragmatics:
                pragmatic_exs = [ex for ex in exs if ex["is_pragmatic"]]
                if len(pragmatic_exs) < args.max_samples:
                    non_pragmatic_exs = [ex for ex in exs if not ex["is_pragmatic"]]
                    diff_len = args.max_samples - len(pragmatic_exs)
                    if len(non_pragmatic_exs) < diff_len:
                        exs = pragmatic_exs + non_pragmatic_exs
                    else:
                        exs = pragmatic_exs + random.sample(non_pragmatic_exs, diff_len)
                else:
                    exs = pragmatic_exs
            
            # Sample primarily examples with "Move the contents of... instructions."
            if args.move_contents:
                mc_exs = [ex for ex in exs if ex["has_move_content"]]
                if len(mc_exs) < args.max_samples:
                    non_mc_exs = [ex for ex in exs if not ex["has_move_content"]]
                    diff_len = args.max_samples - len(mc_exs)
                    if len(non_mc_exs) < diff_len:
                        exs = mc_exs + non_mc_exs
                    else:
                        exs = mc_exs + random.sample(non_mc_exs, diff_len)
                else:
                    exs = mc_exs


            if len(exs) < args.max_samples:
                sample = exs
            else:
                sample = random.sample(exs, args.max_samples)
            
            for ex in sample:
                sample_id = ex["sample_id"]
                global_numops = len(ex["sentence_masked"].rstrip().rstrip(".").split("."))
                if sample_id not in sample_ids:
                    sample_ids[sample_id] = set()
                sample_ids[sample_id].add(global_numops)
        
        with open(args.input_file, encoding="utf-8") as in_f:
            for line in in_f:
                d = json.loads(line)
                if d["sample_id"] in sample_ids:
                    global_numops = len(d["sentence_masked"].rstrip().rstrip(".").split("."))
                    if global_numops in sample_ids[d["sample_id"]]:
                        print(line.strip())

            

            
        
        
            
    
    
    