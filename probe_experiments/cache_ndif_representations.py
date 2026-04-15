import os
import pdb
import csv
import glob
import pickle
import argparse

import pandas as pd

import torch
from torch.utils.data import Dataset
import nnsight
from nnsight import CONFIG, LanguageModel
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from tqdm import tqdm


model_list = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct"
]
# model_name = model_list[-2]


_GPT_MAX_LENGTH = 512
NUM_BOXES = 7

_MAX_SOURCE_TEXT_LENGTH = {
 "t5": 512,
 "gpt": 512,
 "llama": 2048
}

_MAX_TARGET_TEXT_LENGTH = 100

_INPUT_DIMENSIONS = {
 "t5": 768,
 "gpt": 1600,
 "llama": 5120
}


PROMPT = """Given the description after "Description:", write a true statement about a boxes and its contents according to the description after "Statement:".

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.
Statement: Box 3 contains the paper and the string.

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 2 contains the bag and the machine and the map.

Description: """


class InferenceDataset(Dataset):
    """Loads LM dataset for inference."""

    def __init__(self, dataframe, tokenizer, max_length=_GPT_MAX_LENGTH, include_empty=True, condition_on="number",
                  min_prev_objects=-1, include_prompt=False):

         self.tokenizer = tokenizer
         self.include_empty = include_empty
         self.min_prev_objects = min_prev_objects
         self.condition_on = condition_on

         if self.include_empty:
             self.data = dataframe
         else:
             # filter all examples with empty boxes if include_empty is set to False
             f = dataframe["masked_content"].str.contains("nothing") | dataframe["masked_content"].str.contains(
                 "is empty")
             self.data = dataframe[-f]

             if self.min_prev_objects > 0:
                 f = dataframe["masked_content"].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                 self.data = self.data[f]

             self.data = self.data.reset_index()

         self.prefix_text = self.data["prefix"]
         self.target_text = self.data["sentence"]
         self.max_length = max_length
         self.prompt_ends = self.prefix_text[0].split()[-1]  # extract the last token from the prefix
         # prompt ends should be a box number, or "the" or "contains"

         if self.min_prev_objects > 0:
             self.prefix_text = self.data.apply(lambda x: x["prefix"] + " " + " and ".join(
                 x["masked_content"].split(" and ")[0:self.min_prev_objects]) + " and the", axis=1)
         # better logic for constructing prefix, now assumes that the prefix ends with box number

         elif self.condition_on == "contains":
             # add " contains" to prefix
             # extract the last token from the prefix
             if self.prompt_ends.isdigit():
                 self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains")
             elif self.prompt_ends == "contains":
                 pass  # already contains "contains"
         elif self.condition_on == "the":
             # add " contains the" to prefix
             if self.prompt_ends.isdigit():
                 self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains the")
             elif self.prompt_ends == "contains":
                 self.prefix_text = self.data["prefix"].apply(lambda x: x + " the")

         if include_prompt:
             self.prefix_text = self.prefix_text.apply(
                 lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
             self.target_text = self.target_text.apply(
                 lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])

             print(self.prefix_text[0])
             print(self.prefix_text[1])
             print("---------")
             print(self.target_text[1])

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        target_text = str(self.target_text[index])

        targ = self.tokenizer.batch_encode_plus(
         [target_text], max_length=self.max_length, return_tensors='pt', padding="max_length")

        prefix_text = str(self.prefix_text[index])

        pref = self.tokenizer.batch_encode_plus(
         [prefix_text], max_length=self.max_length, return_tensors='pt', padding="max_length")

        target_ids = targ['input_ids'].squeeze()
        prefix_ids = pref['input_ids'].squeeze()
        prefix_attn_masks = pref['attention_mask'].squeeze()

        return {
         'prefix_text': prefix_text,
         'target_ids': target_ids,
         'prefix_ids': prefix_ids,
         'prefix_attn_masks': prefix_attn_masks,
         'target_text': target_text,

        }



def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument("--model_type",
                     default="llama",
                     choices=["t5", "gpt", "llama"],
                     help="'t5', 'gpt' or 'llama' supported.")
    parser.add_argument("--model_name",
                        default="meta-llama/Meta-Llama-3.1-405B",
                        choices=model_list,
                        help=f"{model_list} supported.")
    parser.add_argument("--dataset_path",
                     default="/Users/zhaoqiao/Projects/NLP/entity-tracking/box-models/data/boxes-dataset-v1/few_shot_boxes_nso_exp2_max3",
                     type=str)

    parser.add_argument('--checkpoint_root',
                     default="./probe_checkpoints", type=str)
    parser.add_argument(
         "--object_vocabulary_file",
         type=str,
         default="/Users/zhaoqiao/Projects/NLP/entity-tracking/box-models/data/objects_with_bnc_frequency.csv",
         help='Path to a .csv file with a string field "object_names".')
    parser.add_argument('--layer',
                     default=-1,
                     type=int)
    parser.add_argument('--epo',
                     default=16,
                     type=int)
    parser.add_argument('--condition_on',
                     choices=["box", "period", "the", "number", "contains"],
                     type=str,
                     dest='condition_on',
                     default='the')

    # Non-linear probe
    parser.add_argument('--mid_dim',
                     default=128,
                     type=int)
    parser.add_argument('--twolayer',
                     dest='twolayer',
                     action='store_true',
                     default=True)
    parser.add_argument('--object_location',
                     dest='object_location',
                     action='store_true')
    parser.add_argument('--binary_probe',
                     dest='binary_probe',
                     action='store_true')
    parser.add_argument('--random',
                     dest='random',
                     action='store_true')
    parser.add_argument('--eval_only',
                     dest='eval_only',
                     action='store_true')
    parser.add_argument('--exclude_empty',
                     dest='exclude_empty',
                     action='store_true')
    parser.add_argument('--condition_on_obj',
                     default=0,
                     type=int)

    parser.add_argument('--model_representation_path',
                     default=None,
                     type=str)

    parser.add_argument('--save_model_representation',
                     dest="save_model_representation",
                     default=True,
                     action="store_true")

    parser.add_argument('--load_model_representation',
                     dest="load_model_representation",
                     action="store_true")

    parser.add_argument('--include_prompt',
                     dest="include_prompt",
                     action="store_true")
    parser.add_argument('--ndif_apikey',
                        type=str,
                        required=True)
    parser.add_argument('--hf_token',
                        type=str,
                        required=True)

    args, _ = parser.parse_known_args()
    return args

def main(args):
    CONFIG.API.APIKEY = args.ndif_apikey
    os.environ['HF_TOKEN'] = args.hf_token
    
    # loader_train, loader_test = None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder_name = f"probing/state"

    if args.twolayer:
     folder_name = folder_name + f"_tl{args.mid_dim}"  # tl for probes without batchnorm
    if args.random:
     folder_name = folder_name + "_random"
    if args.object_location:
     folder_name = folder_name + "_object_location"
    if args.binary_probe:
     folder_name = folder_name + "_binary"
    if args.exclude_empty:
     folder_name = folder_name + "_exclude_empty"
    if args.condition_on_obj > 0:
     folder_name = folder_name + f"_condition_on_obj_{args.condition_on_obj}"

    print(f"Running experiment for {folder_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print("[Data]: Reading data...\n")

    # Load data
    data_type = "t5" if args.model_type == "t5" else "gpt"
    dataset_path_train = os.path.join(args.dataset_path, f'train-subsample-states-{data_type}.jsonl')
    print("Train dataset:", dataset_path_train)
    dataset_path_test = os.path.join(args.dataset_path, f'test-subsample-states-{data_type}.jsonl')

    train_df = pd.read_json(dataset_path_train, orient='records', lines=True)
    test_df = pd.read_json(dataset_path_test, orient='records', lines=True)

    if args.eval_only:
        train_df = train_df.head(0)

    if args.model_type == "t5":
        train_df = train_df[["sentence_masked", "masked_content"]]
        test_df = test_df[["sentence_masked", "masked_content"]]

    # Load object names
    object_map = {}
    object_list = []
    with open(args.object_vocabulary_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            object_map[row["object_name"]] = i
            object_list.append(row["object_name"])

    model = LanguageModel(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    act_container_train = []
    act_all_container_train = []
    act_container_test = []
    act_all_container_test = []
    train_dataset = InferenceDataset(train_df, tokenizer, _MAX_SOURCE_TEXT_LENGTH[args.model_type],
                                  include_empty=not args.exclude_empty, condition_on=args.condition_on,
                                  min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt)
    test_dataset = InferenceDataset(test_df, tokenizer, _MAX_SOURCE_TEXT_LENGTH[args.model_type],
                                 include_empty=not args.exclude_empty, condition_on=args.condition_on,
                                 min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt)


    def collate_fn(batch):
         prefix_text = [item['prefix_text'] for item in batch]
         target_text = [item['target_text'] for item in batch]
         prefix_ids = torch.stack([item['prefix_ids'] for item in batch])
         target_ids = torch.stack([item['target_ids'] for item in batch])
         prefix_attn_masks = torch.stack([item['prefix_attn_masks'] for item in batch])

         return [prefix_text, target_text, prefix_ids, target_ids, prefix_attn_masks]

    loader_train = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=8, collate_fn=collate_fn)
    loader_test = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=8, collate_fn=collate_fn)

    # create saving directory
    train_rep_path = os.path.join(args.model_representation_path, 'representations_train.p')
    test_rep_path = os.path.join(args.model_representation_path, 'representations_test.p')

    end_idx = None
    # I've checked that " n" between 0 and 7 are all single tokens.
    # But this is probably a patchy solution if box num >= 8
    if args.condition_on == "number":
        # end_idx = -1 # remove the last token which is a number
        raise NotImplementedError("Logics for fewshot dataset not implemented")
    elif args.condition_on == "box":
        end_idx = -2  # not -1 because there's a 'contains' token in the fewshot data
        raise NotImplementedError("Logics for fewshot dataset not implemented")
    # "Box" is one token
    elif args.condition_on == "period":
        end_idx = -3  # not -2, same reason
        raise NotImplementedError("Logics for fewshot dataset not implemented")

    # Used some patchy logics in the dataset code to make sure that when condition_on is "the" or "contains", it is corresponds to the last token so end_idx is None and we save the activation of the last token

    # for condition on == the or contains, it is specified in the dataset and corresponds to the last token
    # with model.session(remote=True) as session: # too many data cannot be computed in one session
    num_invoked = 0
    for rep_path, loader in zip([train_rep_path, test_rep_path],[loader_train, loader_test]):
        print(f"caching {rep_path}...")
        for idx, data in enumerate(tqdm(loader)):
             # check if current batch has been processed
             batch_representation_path = rep_path.replace(".p", f"_{idx}.p")
             if os.path.exists(batch_representation_path):
                 print(f"Batch {idx} already processed, skipping...")
                 continue

             with model.session(remote=True) as session:
                 _act_all_container_train = nnsight.list().save()
                 _act_container_train = nnsight.list().save()
                 prefix_text, target_text, prefix_ids, target_ids, prefix_attn_masks = data
                 if end_idx is not None:
                     prefix_ids = prefix_ids[:, :end_idx]
                     prefix_attn_masks = prefix_attn_masks[:, :end_idx]
                 with model.trace({
                     "input_ids": prefix_ids,
                     'attention_mask': prefix_attn_masks,
                 }, remote=True) as tracer:
                     if args.save_model_representation:
                         stacked_tensor = torch.cat(
                             [layer.output[0][:, -1, :].detach().cpu().unsqueeze(0) for layer in model.model.layers],
                             dim=0
                         ).permute(1, 0, 2)
                         # stacked_tensor = torch.cat([tensor.unsqueeze(0) for tensor in layer_tensors], dim=0)[:,:,-1,:].permute(1,0,2)
                         _act_all_container_train.append(stacked_tensor)
                         last_hs = model.model.layers[args.layer - 1].output[0][:, -1, :].detach().cpu()
                         _act_container_train.append(last_hs)
                     else:
                         last_hs = model.model.layers[-1].output[0][:, -1, :].detach().cpu()
                         _act_container_train.append(last_hs)

             # save the representation for the current batch
             if args.save_model_representation:
                 with open(batch_representation_path, "wb") as rep_f:
                     pickle.dump(_act_all_container_train, rep_f)
                     _act_all_container_train.clear()
             print(f"Batch {idx} processed, saved to {batch_representation_path}")

    # concatenate batched files 
    print("aggregating all files now...")
    for split, rep_path in zip(["train", "test"], [train_rep_path, test_rep_path]):
        files = glob.glob(os.path.join(rep_path, f"representations_{split}_*.p"))
        split_data = []
        with open(files, "rb") as rep_f:
            data = pickle.load(rep_f)
            split_data.append(data[0])  # B, L, H
            del data

        # data: num_batch , batch_size, layer_num, hidden_size
        split_data = torch.cat(split_data, dim=0).permute(1, 0, 2)  # L, B, H
        file_name = f'representations_{split}.p'
        file_path = os.path.join(rep_path, file_name)
        with open(file_path, "wb") as rep_f:
            pickle.dump(split_data, rep_f)
            del split_data

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    args = parse_args()
    main(args)
