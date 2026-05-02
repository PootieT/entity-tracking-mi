import argparse
import json
import pandas as pd
import re

NUM_BOXES = 7

def convert_probe_predictions(probe_output_path, dataset_path, binary=False, only_mentioned=False):
    targets = []
    predictions = []
    contexts = []
    
    
    with open(probe_output_path, "r") as in_f, open(dataset_path, "r") as data_f:
        object_names = in_f.readline().strip().split(" ")
        for i, line in enumerate(in_f):
            
            if i % NUM_BOXES == 0 and not binary:
                local_targets = []
                local_contexts = []
                local_predictions = {}
                for b in range(NUM_BOXES):
                    d = json.loads(data_f.readline())
                    local_targets.append(d["masked_content"].replace("<extra_id_0> ", ""))
                    local_contexts.append(d["sentence_masked"].replace("<extra_id_0> ", ""))
                    local_predictions[b] = []
            elif binary:
                d = json.loads(data_f.readline())
 
            mentioned_objects = set()
            if only_mentioned:
                m = re.findall("the [^ ,.]+", d["sentence_masked"])
                mentioned_objects.update([x.replace("the ", "") for x in m])
            
            locations = [int(l) for l in line.strip().split(" ")]
            if not binary:
                boxes = [set() for _ in range(NUM_BOXES)]
                for j, l in enumerate(locations):
                    if l > 0 and (not only_mentioned or object_names[j] in mentioned_objects):
                        boxes[l-1].add(f"the {object_names[j]}")
            else:
                box_contents = set()
                for j, l in enumerate(locations):
                    if l > 0 and (not only_mentioned or object_names[j] in mentioned_objects):
                        box_contents.add(f"the {object_names[j]}")
            

            if not binary:
                for b in range(NUM_BOXES):
                    if len(boxes[b]) > 0:
                        pred = "contains " + " and ".join(boxes[b])
                    else:
                        pred = "is empty"
                    local_predictions[b].append(pred)
            
            if not binary and i % NUM_BOXES == (NUM_BOXES - 1):
                for b in range(NUM_BOXES):
                    for pred in local_predictions[b]:
                        targets.append(local_targets[b])
                        predictions.append(pred)
                        contexts.append(local_contexts[b])

            if binary:
                targets.append(d["masked_content"].replace("<extra_id_0> ", ""))
                contexts.append(d["sentence_masked"].replace("<extra_id_0> ", ""))
                if len(box_contents) > 0:
                    pred = "contains " + " and ".join(box_contents)
                else:
                    pred = "is empty"
                predictions.append(pred)
  

    final_df = pd.DataFrame({'target': targets, 'prediction': predictions,'input': contexts})
    return final_df

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--probe_output", type=str)
    argparser.add_argument("--dataset", type=str)

    argparser.add_argument("--output", type=str)
    argparser.add_argument("--binary", action="store_true", dest="binary")
    
    args = argparser.parse_args()
    
    final_df = convert_probe_predictions(args.probe_output, args.dataset, binary=args.binary)
    final_df.to_csv(args.output, sep='\t')
 
if __name__ == "__main__":
    main()
