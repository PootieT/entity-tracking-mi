import argparse
import collections
import json


def compute_intervention_metrics(output_path):
    correct = 0
    total = 0
    collapsed = 0
    with open(output_path) as f:
        for line in f:
            d = json.loads(line)
            model_answer = d["intervened_answer"].strip().rstrip(".")
            if model_answer.startswith("</s>"):
                model_items = []
                collapsed += 1
            else:
                model_items = model_answer.lower().split(" and ")
                model_items = [i.removeprefix("the ") for i in model_items]
            print("Model answer: ", model_answer)
            print("Model items: ", model_items)
            target_items = d["orig_answer"]
            intervention_target_item = d["intervention_target_item"]
            intervention_operation = d["intervention_operation"]

            if intervention_operation == "add":
                target_items.append(intervention_target_item)
            elif intervention_operation == "remove":
                target_items.remove(intervention_target_item)
            
            print(f"Intervention item ({intervention_operation}): ", intervention_target_item)
            print("Target items: ", target_items)
            print()
            if collections.Counter(model_items) == collections.Counter(target_items):
                correct += 1
            total += 1
    print(correct/total, correct, total)

def main():
    """
        Command line utility.
    """

    parser = argparse.ArgumentParser(
        "CLI utility for evaluating model outputs.")
    parser.add_argument(
        "--model_output", type=str, required=True,
        help="Path to model output in jsonl format.",
    )
    args = parser.parse_args()
    compute_intervention_metrics(args.model_output)


if __name__ == "__main__":
    main()

