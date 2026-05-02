import os
# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
import time
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from src.dataset import ProbeDataLoader, LMDataloader, GPTDataloaderForInference, ObjectLocationProbeDataLoader, BinaryProbeDataLoader, MentionedProbeDataLoader, GPTDataloaderForIncrementalLocalState, IncrementalLocalStateProbeDataLoader
from src.model import T5ForProbing, GPTForProbing, LlamaForProbing
from src.probe_trainer import Trainer, TrainerConfig, Mention_Trainer
from src.probe_model import BatteryProbeClassification, ObjectLocationProbeClassification, BatteryProbeClassificationTwoLayer
from transformers import AutoTokenizer
import pickle
from nnsight import LanguageModel
from src.ndif_adapt import save_model_activations, setup_ndif_api, save_activations_ckpt
from transformers import BitsAndBytesConfig

# load quantization config


_MAX_SOURCE_TEXT_LENGTH = {
    "t5": 512,
    "gpt": 512,
    # "llama": 2048
    "llama": 512
}
# 512 is sufficient 

_MAX_TARGET_TEXT_LENGTH = 100


# make deterministic
torch.manual_seed(0)


def main():
    
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument("--model_type",
                        required=True,
                        choices=["t5", "gpt", "llama"],
                        help="'t5', 'gpt' or 'llama' supported.")
    parser.add_argument("--dataset_path",
                        required=True,
                        type=str)
    parser.add_argument("--model_path",
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('--checkpoint_root',
                        default="./probe_checkpoints", type=str)
    parser.add_argument(
            "--object_vocabulary_file",
            type=str,
            default="data/objects_with_bnc_frequency.csv",
            help='Path to a .csv file with a string field "object_names".')
    parser.add_argument('--layer',
                        required=True,
                        default=-1,
                        type=int)
    parser.add_argument('--epo',
                        default=16,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='Learning rate for the optimizer.')
    
    
    parser.add_argument('--condition_on', 
                        choices=["box", "period", "the", "number", "contains", "statement", "last_box"],
                        type=str,
                        dest='condition_on',
                        default='number')
    
    
    parser.add_argument('--exp_name', choices=["binary", "mentioned", "global", "incremental_local_state"], type=str, default="binary",)

    # Non-linear probe
    parser.add_argument('--mid_dim',
                        default=128,
                        type=int)
    parser.add_argument('--twolayer',
                        dest='twolayer', 
                        action='store_true')
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
                        action="store_true")
    
    parser.add_argument('--load_model_representation',
                        dest="load_model_representation",
                        action="store_true")

    parser.add_argument('--include_prompt',
                        dest="include_prompt",
                        action="store_true")
    
    parser.add_argument('--subsample',
                        help="If use subsampled dataset",
                        action="store_true")
    parser.add_argument('--debug', 
                        help="If debug mode",
                        action="store_true")
    
    
    parser.add_argument('--remote', 
                        help="If use remote server for nnsight",
                        action="store_true") # if using remote, then definitely using nnsight models
    
    parser.add_argument('--act_batch_size',
                        default=1,
                        type=int,
                        help="Batch size for remote model inference, only used when --remote is set")
    
    parser.add_argument( '--save_activations_ckpts',
                        default=False,
                        action='store_true',
                        help="Whether to save the model activations per batch and merge later. It should be used when using NDIF models -- they might get blocked we should resume from the last saved batch"
    
    )
    parser.add_argument("--num_samples_to_use",
                        default=None,
                        type=int,
                        help="For debugging NDIF, only use a subset of the data")
    parser.add_argument('--quant', 
                        help="If use quantized model for NDIF",
                        action="store_true")
    parser.add_argument('--use_altform',
                        action="store_true",
                        help="Whether using altform dataset, just used to control the choice of prompt examples.")


    args, _ = parser.parse_known_args()
    setup_ndif_api()

    if (args.condition_on not in ["number", "contains"]) and args.model_type == 't5':
        raise ValueError("--condition_on must be set to 'number' or 'contains' when training a probe on T5.")
    if args.eval_only:
    
        raise ValueError("--eval_only is buggy, do not use for now")

    if args.exclude_empty and not args.binary_probe:
        raise ValueError("--exclude_empty only works with --binary_probe")

    # if args.exclude_empty and args.condition_on not in ["contains", "the"]:
    #     raise ValueError("--exclude_empty can only be used with --condition_on 'contains' or 'the'")
    
    # if args.condition_on in ["contains", "the"] and not args.exclude_empty:
    #     raise ValueError("--condition_on 'contains' or 'the' can only be used with --exclude_empty")
    
    if args.exp_name == "incremental_local_state" and args.condition_on != "number":
        raise ValueError("exp_name 'incremental_local_state' requires --condition_on 'number'")
    
    if args.save_model_representation and args.model_representation_path is None:
        raise ValueError("--save_model_representation requires --model_representation_path to be set")
    
    if args.load_model_representation and args.model_representation_path is None:
        raise ValueError("--load_model_representation requires --model_representation_path to be set")
    
    if args.load_model_representation and args.save_model_representation:
        raise ValueError("--load_model_representation and --save_model_representation cannot be used together")

    if not args.include_prompt and args.condition_on == "statement":
        raise ValueError("--condition_on 'statement' requires --include_prompt to be set")
    
    # check remote setting
    
    
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
    if args.subsample:
        folder_name = folder_name + "_subsample"
        
    if not args.include_prompt:
        folder_name = folder_name + "_no_prompt"
        

    print(f"Running experiment for {folder_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print("[Data]: Reading data...\n")

    # Load data
    print("args.model_type:", args.model_type)
    data_type = "t5" if args.model_type == "t5" else "gpt" 
    if args.subsample:
        dataset_path_train = os.path.join(args.dataset_path, f'train-subsample-states-{data_type}.jsonl')
    else:
        dataset_path_train = os.path.join(args.dataset_path, f'train-{data_type}.jsonl')
    print("Train dataset:", dataset_path_train)
    if args.subsample:
        dataset_path_test = os.path.join(args.dataset_path, f'test-subsample-states-{data_type}.jsonl')
    else:
        dataset_path_test = os.path.join(args.dataset_path, f'test-{data_type}.jsonl')
    
    
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

    act_container_train = []
    act_all_container_train = []
    act_container_test = []
    act_all_container_test = []



    if args.load_model_representation:
        
        # for debugging:
        if args.debug:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            train_dataset = GPTDataloaderForIncrementalLocalState(train_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
            test_dataset = GPTDataloaderForIncrementalLocalState(test_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
            act_container_train = train_dataset.build_dummy_activations()
            act_container_test = test_dataset.build_dummy_activations() 
        
        else:
            print("Loading pre-computed model representations from:", args.model_representation_path)
            # load pre-computed representations
            # super messy. just to make it compatible with different formats of saved representations
            # pdb.set_trace(header="debug_load")
            files = os.listdir(args.model_representation_path)
            use_torch = any(f.endswith(".pt") for f in files)
            if use_torch:
                train_rep_path = os.path.join(args.model_representation_path, "representations_train.pt")
                test_rep_path = os.path.join(args.model_representation_path, "representations_test.pt")
            else:
                train_rep_path = os.path.join(args.model_representation_path, "representations_train.p")
                test_rep_path = os.path.join(args.model_representation_path, "representations_test.p")
            # pdb.set_trace(header="debuging load act")
            with open(train_rep_path, "rb") as rep_f:
                if use_torch:
                    act_all_container_train = torch.load(rep_f)
                else:
                    
                    act_all_container_train = pickle.load(rep_f)
            # print(f"shape of act_all_container_train: {len(act_all_container_train)}, each of shape {len(act_all_container_train[0])}")
            
            
            if type(act_all_container_train) is list:
                # When using codellama
                for act in act_all_container_train:
                    act_container_train.append(act[args.layer - 1])
                    # print(f"each act_container_train shape: {act[args.layer - 1].shape}")
                act_all_container_train.clear()
            elif type(act_all_container_train) is torch.Tensor:
                # Just used for the old activation files transferred from nnsight local-dev branch
                # For the remote-executed ones, they should be lists of tensors already
                act_container_train = act_all_container_train.permute(1,0,2)[args.layer - 1].to(torch.float32)
                del act_all_container_train
            
            

            with open(test_rep_path, "rb") as rep_f:
                if use_torch:
                    act_all_container_test = torch.load(rep_f)
                else:
                    act_all_container_test = pickle.load(rep_f)

            # print(f"shape of act_all_container_test: {len(act_all_container_test)}, each of shape {len(act_all_container_test[0])}")
            if type(act_all_container_test) is list:
                # When using codellama
                for act in act_all_container_test:
                    act_container_test.append(act[args.layer - 1])
                act_all_container_test.clear()
            elif type(act_all_container_test) is torch.Tensor:
                act_container_test = act_all_container_test.permute(1,0,2)[args.layer - 1].to(torch.float32)
                del act_all_container_test

    else:
        if not args.debug:
            # Load T5 model to compute representations
            if args.quant:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
            else:
                bnb_config = None
            if not args.remote: 
                if args.model_type == "t5": 
                    model = T5ForProbing.from_pretrained(args.model_path)
                elif args.model_type == "gpt":
                    model = GPTForProbing.from_pretrained(args.model_path)
                elif args.model_type == "llama":
                    model = LlamaForProbing.from_pretrained(args.model_path, quantization_config=bnb_config)

                # Set probe layer (1-indexed)
                model.probe_layer = args.layer
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                if not args.quant:
                    model.to(device)
                model.eval()
            else:
                model = LanguageModel(args.model_path, device_map="auto")
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = None
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            

        # initialze LM test dataset
        if args.model_type == "t5":
            train_dataset = LMDataloader(train_df, tokenizer, _MAX_SOURCE_TEXT_LENGTH[args.model_type], _MAX_TARGET_TEXT_LENGTH, "sentence_masked", "masked_content", include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
            test_dataset = LMDataloader(test_df, tokenizer, _MAX_SOURCE_TEXT_LENGTH[args.model_type], _MAX_TARGET_TEXT_LENGTH, "sentence_masked", "masked_content", include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
        elif args.model_type in ["gpt", "llama"]:
            # few shot data ends with contains but t5 data ends with box number
            
            if args.exp_name == "incremental_local_state":
                train_dataset = GPTDataloaderForIncrementalLocalState(train_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
                test_dataset = GPTDataloaderForIncrementalLocalState(test_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
            else: 
                max_length = 256 if not args.include_prompt else _MAX_SOURCE_TEXT_LENGTH[args.model_type]
                train_dataset = GPTDataloaderForInference(train_df, tokenizer, max_length=max_length, include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, is_altform=args.use_altform)
                test_dataset = GPTDataloaderForInference(test_df, tokenizer, max_length=max_length, include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, is_altform=args.use_altform)
                
        if args.num_samples_to_use is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, list(range(args.num_samples_to_use)))
            test_dataset = torch.utils.data.Subset(test_dataset, list(range(args.num_samples_to_use)))
            

        loader_train = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=args.act_batch_size, num_workers=1)
        loader_test = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=args.act_batch_size, num_workers=1)

        if not (args.exp_name in ["incremental_local_state"]):
            num_train_batches = len(loader_train)
            print(f"Number of batches in training set: {num_train_batches}")
            cache_ckpt_dir = os.path.join(args.model_representation_path, "ckpts")
            train_ckpt_dir = os.path.join(cache_ckpt_dir, "train")
            test_ckpt_dir = os.path.join(cache_ckpt_dir, "test")
            if args.save_activations_ckpts:
                if not os.path.exists(train_ckpt_dir):
                    os.makedirs(train_ckpt_dir)
                    num_completed_batches_train = 0
                    # count all the files in the dir
                else:
                    num_completed_batches_train = len([name for name in os.listdir(train_ckpt_dir) if os.path.isfile(os.path.join(train_ckpt_dir, name))])
                if not os.path.exists(test_ckpt_dir):
                    os.makedirs(test_ckpt_dir)
                    num_completed_batches_test = 0
                else:
                    num_completed_batches_test = len([name for name in os.listdir(test_ckpt_dir) if os.path.isfile(os.path.join(test_ckpt_dir, name))])
                

            if args.model_type == "t5":
            # This brach is deprecated since we are not using T5 anymore, but just keep it here for now.   
                

                the_token = torch.tensor(8, dtype=torch.long)

                for data in tqdm(loader_train, total=len(loader_train)):
                    token_idx = 0
                    the_pos = 0

                    if args.condition_on == "contains":
                        token_idx = 1
                    elif args.condition_on == "the":
                        token_idx = 2
                        the_pos = 1

                
                    while args.condition_on_obj >= the_pos:
                        # 'the' has the token index 8 for T5
                        token_idx = list(data['target_ids'][0]).index(the_token, token_idx + 1)
                        the_pos +=1

                    decoder_input = data['target_ids'][:,0:(token_idx+1)].to(device, dtype=torch.long)
                    ids = data['source_ids'].to(device, dtype=torch.long)
                    mask = data['source_mask'].to(device, dtype=torch.long)

                    #print(tokenizer.convert_ids_to_tokens(decoder_input[0]))

                    if args.save_model_representation:
                        act = model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder_input, return_all_layers=True) #representation at first (=mask) token
                        act_container_train.append(act[args.layer - 1][0,-1,:].detach().cpu())
                        act_all_container_train.append([a[0,-1,:].detach().cpu() for a in act])
                    else:
                        # forward function automatically outputs the representation at `layer`
                        act = model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder_input)[0,-1,:].detach().cpu() #representation at first (=mask) token
                        act_container_train.append(act)

   
                
                for data in tqdm(loader_test, total=len(loader_test)):

                    token_idx = 0
                    the_pos = 0

                    if args.condition_on == "contains":
                        token_idx = 1
                    elif args.condition_on == "the":
                        token_idx = 2
                        the_pos = 1

                
                    while args.condition_on_obj >= the_pos:
                        # 'the' has the token index 8 for T5
                        token_idx = list(data['target_ids'][0]).index(the_token, token_idx + 1)
                        the_pos +=1


                    decoder_input = data['target_ids'][:,0:(token_idx+1)].to(device, dtype=torch.long)
                    ids = data['source_ids'].to(device, dtype=torch.long)
                    mask = data['source_mask'].to(device, dtype=torch.long)

                    # print(tokenizer.convert_ids_to_tokens(labels[0][:6]))

                    if args.save_model_representation:
                        act = model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder_input, return_all_layers=True) #representation at first (=mask) token
                        act_container_test.append(act[args.layer - 1][0,-1,:].detach().cpu())
                        act_all_container_test.append([a[0,-1,:].detach().cpu() for a in act])
                    else:
                        # forward function automatically outputs the representation at `layer`
                        act = model(input_ids=ids, attention_mask=mask, decoder_input_ids=decoder_input)[0,-1,:].detach().cpu() #representation at first (=mask) token
                        act_container_test.append(act)
                    
                    # Activate this for debugging on a handful of examples
                    #if len(act_container_test) == 70:
                    #     break
                    
            elif args.model_type in ["gpt", "llama"]:
                # pdb.set_trace(header="checking end idx")
                end_idx = None
                # I've checked that " n" between 0 and 7 are all single tokens.
                # But this is probably a patchy solution if box num >= 8        
                if args.condition_on == "box":
                    end_idx = -2
                # "Box" is one token
                elif args.condition_on == "number":
                    end_idx = None # last token is already the box number
                elif args.condition_on == "period": # fails with include_prompt=True
                    
                    if args.include_prompt:
                        end_idx = -6
                    end_idx = -3
                elif args.condition_on == "last_box":
                    end_idx = -4 # need to make sure not used with include_prompt=True by urself.
                    
                if args.condition_on == "statement":
                    end_idx = -4
                
                for idx, data in tqdm(enumerate(loader_train), total=len(loader_train)):
                    
                    if args.save_activations_ckpts and idx < num_completed_batches_train:
                        # skip the processed batches, for simplicity we assume the batch size is not changed over different runs
                        continue
                    ids = data['prefix_ids'].to(device, dtype=torch.long)
                    # assert tokenizer.convert_ids_to_tokens(ids[0][:end_idx])[-1] == "."
                    # continue
                    mask = data['prefix_attn_masks'].to(device, dtype=torch.long)
                    if end_idx is not None:
                        ids = ids[0][:end_idx].unsqueeze(0)
                        mask = mask[0][:end_idx].unsqueeze(0)
                    if args.save_model_representation:
                        if not args.remote:
                            act = model(input_ids=ids, attention_mask=mask, return_all_layers=True)
                            act_all_container_train.append([a[0,-1,:].detach().cpu() for a in act])
                            
                            act_container_train.append(act[args.layer - 1][0,-1,:].detach().cpu())
                        else:
                            # TODO examine the output shapes closely here. 
                            # pdb.set_trace(header="debug")
                            num_layers = len(model.model.layers) + 1 # plus input embeddding
                            hidden_dim = model.model.layers[0].mlp.down_proj.out_features # assuming llama3 model
                            with model.trace(
                                {
                                    "input_ids": ids,
                                    "attention_mask": mask,
                                },
                                remote=True
                            ) as tracer:
                                stacked_hs = torch.zeros((num_layers, ids.size(0), hidden_dim)).save() # need to be variable,
                                stacked_hs[0,:,:] = model.model.layers[0].input[:, -1, :].detach().cpu()  # input of the first layer
                                for idx, layer in enumerate(model.model.layers):
                                    stacked_hs[idx + 1, :, :] = layer.output[:, -1, :].detach().cpu()
                                    
                            act = stacked_hs
                                    
                                
                            # maybe permute to b, n_layers, hidden_dim, and extend
                            act = act.permute(1,0,2).to(torch.float32) # batch, n_layers, hidden_dim
                            # convert to a list of tensors, of shape, num_samples, num_layers, hidden_dim
                            act = [act[i] for i in range(act.size(0))]
                            act_all_container_train.extend(act)
                            
                            # TODO save ckpt if need
                            if args.save_activations_ckpts:
                                save_activations_ckpt(act, train_ckpt_dir, idx)
                            
                                            

                    else:              
                        # last hidden state
                        act = model(input_ids=ids, attention_mask=mask)[0,-1,:].detach().cpu()
                        act_container_train.append(act)

                # pdb.set_trace(header="checking test set")
                for idx, data in tqdm(enumerate(loader_test), total=len(loader_test)):
                    if args.save_activations_ckpts and idx < num_completed_batches_test:
                        # skip the processed batches, for simplicity we assume the batch size is not changed over different runs
                        continue
                    
                    
                    ids = data['prefix_ids'].to(device, dtype=torch.long)
                    mask = data['prefix_attn_masks'].to(device, dtype=torch.long)
                    
                    if end_idx is not None:
                        ids = ids[0][:end_idx].unsqueeze(0)
                        mask = mask[0][:end_idx].unsqueeze(0)
                    # last hidden state
                    if args.save_model_representation:
                        if not args.remote:
                            act = model(input_ids=ids, attention_mask=mask, return_all_layers=True)
                            act_all_container_test.append([a[0,-1,:].detach().cpu() for a in act])
                            act_container_test.append(act[args.layer - 1][0,-1,:].detach().cpu())            
                        else: 
                            num_layers = len(model.model.layers) + 1 # plus input embeddding
                            hidden_dim = model.model.layers[0].mlp.down_proj.out_features # assuming llama3 model
                            with model.trace(
                                {
                                    "input_ids": ids,
                                    "attention_mask": mask,
                                },
                                remote=True
                            ) as tracer:
                                stacked_hs = torch.zeros((num_layers, ids.size(0), hidden_dim)).save() # need to be variable,
                                stacked_hs[0,:,:] = model.model.layers[0].input[:, -1, :].detach().cpu()  # input of the first layer
                                for idx, layer in enumerate(model.model.layers):
                                    stacked_hs[idx + 1, :, :] = layer.output[:, -1, :].detach().cpu()
                            act = stacked_hs
                            act = act.permute(1,0,2).to(torch.float32) # batch, n_layers, hidden_dim
                            act = [act[i] for i in range(act.size(0))]
                            if args.save_activations_ckpts:
                                save_activations_ckpt(act, test_ckpt_dir, idx)

                    else:              
                        # last hidden state
                        act = model(input_ids=ids, attention_mask=mask)[0,-1,:].detach().cpu()
                        act_container_test.append(act)        
        else:
            # incremental local state only, supposed to be used with llama/gpt model only
            assert args.model_type in ["gpt", "llama"] and args.condition_on == "number"
            end_idx = None # default ends with the box number token
            
            cache_ckpt_dir = os.path.join(args.model_representation_path, "ckpts")
            train_ckpt_dir = os.path.join(cache_ckpt_dir, "train")
            test_ckpt_dir = os.path.join(cache_ckpt_dir, "test")
            
            if args.save_activations_ckpts:
                if not os.path.exists(train_ckpt_dir):
                    os.makedirs(train_ckpt_dir)
                    num_completed_batches_train = 0
                    # count all the files in the dir
                else:
                    num_completed_batches_train = len([name for name in os.listdir(train_ckpt_dir) if os.path.isfile(os.path.join(train_ckpt_dir, name))])
                if not os.path.exists(test_ckpt_dir):
                    os.makedirs(test_ckpt_dir)
                    num_completed_batches_test = 0
                else:
                    num_completed_batches_test = len([name for name in os.listdir(test_ckpt_dir) if os.path.isfile(os.path.join(test_ckpt_dir, name))])
                
            for idx, data in tqdm(enumerate(loader_train), total=len(loader_train)):
                
                num_train_batches = len(loader_train)
                print(f"Number of batches in training set: {num_train_batches}")
                if args.save_activations_ckpts and idx < num_completed_batches_train:
                    # skip the processed batches, for simplicity we assume the batch size is not changed over different runs
                    continue
                
                ids = data['prefix_ids'].to(device, dtype=torch.long)
                mask = data['prefix_attn_masks'].to(device, dtype=torch.long)
                box_id_positions = data['box_id_positions_flattened']
                if end_idx is not None:
                    ids = ids[0][:end_idx].unsqueeze(0)
                    mask = mask[0][:end_idx].unsqueeze(0)
                if args.save_model_representation:
                    if not args.remote:
                        act = model(input_ids=ids, attention_mask=mask, return_all_layers=True) # shape: [N_Layers + 1 (batch norm layers), batch, seq_len, hidden_dim]
                        act_all_container_train.append([a[0, box_id_positions, :].detach().cpu() for a in act]) # should be a list of N+1 Layers, each of shape [N_box_ids, hidden_dim] ?    
                        act_container_train.append(act[args.layer - 1][0,box_id_positions,:].detach().cpu())
                    else:
                        time.sleep(10) # just make sure not ddosing the server lol
                        
                        num_layers = len(model.model.layers) + 1 # plus input embeddding
                        hidden_dim = model.model.layers[0].mlp.down_proj.out_features # assuming llama3 model
                        with model.trace(
                            {
                                "input_ids": ids,
                                "attention_mask": mask,
                            },
                            remote=True
                        ) as tracer:
                            stacked_hs = torch.zeros((num_layers, len(box_id_positions), hidden_dim)).save() # need to be variable,
                            stacked_hs[0,:,:] = model.model.layers[0].input[:, box_id_positions, :].detach().cpu()  # input of the first layer
                            for idx, layer in enumerate(model.model.layers):
                                stacked_hs[idx + 1, :, :] = layer.output[:, box_id_positions, :].detach().cpu()
                        act = stacked_hs
                        act = act.permute(1,0,2).to(torch.float32) # n_box_ids, n_layers, hidden_dim
                        act = [act[i] for i in range(act.size(0))]
                        
                        
                        if args.save_activations_ckpts:
                            save_activations_ckpt(act, train_ckpt_dir, idx)

                else:              
                    # last hidden state
                    act = model(input_ids=ids, attention_mask=mask)[0,box_id_positions,:].detach().cpu()
                    act_container_train.append(act)


            for idx, data in tqdm(enumerate(loader_test), total=len(loader_test)):
                
                if args.save_activations_ckpts and idx < num_completed_batches_test:
                    # skip the processed batches, for simplicity we assume the batch size is not changed over different runs
                    continue
                time.sleep(10) # just make sure not ddosing the server lol
                ids = data['prefix_ids'].to(device, dtype=torch.long)
                mask = data['prefix_attn_masks'].to(device, dtype=torch.long)
                box_id_positions = data['box_id_positions_flattened']
                if end_idx is not None:
                    ids = ids[0][:end_idx].unsqueeze(0)
                    mask = mask[0][:end_idx].unsqueeze(0)
                # last hidden state
                if args.save_model_representation:
                    if not args.remote:
                            
                        act = model(input_ids=ids, attention_mask=mask, return_all_layers=True)
                        # finally, should be a list of N_Samples, each is a list of N_Layers, each is a tensor of shape [N_box_ids, hidden_dim]
                        act_all_container_test.append([a[0, box_id_positions, :].detach().cpu() for a in act])
                        act_container_test.append(act[args.layer - 1][0,box_id_positions,:].detach().cpu())
                    else:
                        # TODO examine the output shapes closely here.
                        num_layers = len(model.model.layers) + 1 # plus input embeddding
                        hidden_dim = model.model.layers[0].mlp.down_proj.out_features # assuming llama3 model
                        with model.trace(
                            {
                                "input_ids": ids,
                                "attention_mask": mask,
                            },
                            remote=True
                        ) as tracer:
                            stacked_hs = torch.zeros((num_layers, len(box_id_positions), hidden_dim)).save() 
                            stacked_hs[0,:,:] = model.model.layers[0].input[:, box_id_positions, :].detach().cpu()  # input of the first layer
                            for idx, layer in enumerate(model.model.layers):
                                stacked_hs[idx + 1, :, :] = layer.output[:, box_id_positions, :].detach().cpu()
                        act = stacked_hs
                        act = act.permute(1,0,2).to(torch.float32) # n_box_ids, n_layers, hidden_dim
                        act = [act[i] for i in range(act.size(0))]
                        act_all_container_test.extend(act)
                        if args.save_activations_ckpts:
                            save_activations_ckpt(act, test_ckpt_dir, idx)
                    
                else:              
                    # last hidden state
                    act = model(input_ids=ids, attention_mask=mask)[0,box_id_positions,:].detach().cpu()
                    act_container_test.append(act)

        if args.save_model_representation:
            if not os.path.exists(args.model_representation_path):
                os.makedirs(args.model_representation_path)
            # pdb.set_trace(header="db")
            
            
            # check if cached file exists already
            if args.save_activations_ckpts:
                # pdb.set_trace(header="debug save")
                act_all_container_train = []
                act_all_container_test = []
                num_train_ckpts = len(os.listdir(train_ckpt_dir))
                num_test_ckpts = len(os.listdir(test_ckpt_dir))
                for i in range(num_train_ckpts):
                    f = f"batch_{i}.pt"
                    if f.endswith(".pt"):
                        dat = torch.load(os.path.join(train_ckpt_dir, f))
                        if not args.exp_name == "incremental_local_state":
                            act_all_container_train.extend(dat)
                        else:
                            # we actually need to make it [bs*t[nbx,h]]
                            dat = torch.stack(dat).permute(1,0,2)
                            dat = [d for d in dat]
                            act_all_container_train.append(dat) # for incremental local state, each dat is already a list of tensors of shape [N_box_ids, hidden_dim], we don't need to further extend it.
                for i in range(num_test_ckpts):
                    f = f"batch_{i}.pt"
                    if f.endswith(".pt"):
                        dat = torch.load(os.path.join(test_ckpt_dir, f))
                        if not args.exp_name == "incremental_local_state":
                            act_all_container_test.extend(dat)
                        else:
                            dat = torch.stack(dat).permute(1,0,2)
                            dat = [d for d in dat]
                            act_all_container_test.append(dat) # for incremental local state, each dat is already a list of tensors of shape [N_box_ids, hidden_dim], we don't need to further extend it.   
            
            f_train_name = "representations_train.p" if not args.save_activations_ckpts else "representations_train.pt"
            f_test_name = "representations_test.p" if not args.save_activations_ckpts else "representations_test.pt"
            train_rep_path = os.path.join(args.model_representation_path, f_train_name)
            test_rep_path = os.path.join(args.model_representation_path, f_test_name)
            print("Saving model representations to:", train_rep_path, " and ", test_rep_path)   
            with open(train_rep_path, "wb") as rep_f:
                if f_train_name.endswith(".p"):
                    pickle.dump(act_all_container_train, rep_f)
                elif f_train_name.endswith(".pt"):
                    torch.save(act_all_container_train, rep_f)
                act_all_container_train.clear()
            with open(test_rep_path, "wb") as rep_f:
                if f_test_name.endswith(".p"):
                    pickle.dump(act_all_container_test, rep_f)
                elif f_test_name.endswith(".pt"):
                    torch.save(act_all_container_test, rep_f)
                    
                act_all_container_test.clear()
            


    if args.exp_name == "ternary":
        probe_class = 3
    elif args.binary_probe:
        probe_class = 2
    else: 
        probe_class = 8
    
        
        
    # pdb.set_trace(header="debug build ds")
    if args.exp_name == "mentioned":
        probing_dataset_train = MentionedProbeDataLoader(act_container_train, dataset_path_train, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
        probing_dataset_test = MentionedProbeDataLoader(act_container_test, dataset_path_test, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
    elif args.exp_name == "binary":
        probing_dataset_train = BinaryProbeDataLoader(act_container_train, dataset_path_train, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
        probing_dataset_test = BinaryProbeDataLoader(act_container_test, dataset_path_test, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
    elif args.object_location:
        probing_dataset_train = ObjectLocationProbeDataLoader(act_container_train, dataset_path_train, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
        probing_dataset_test = ObjectLocationProbeDataLoader(act_container_test, dataset_path_test, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
    elif args.exp_name == "incremental_local_state":
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        train_dataset = GPTDataloaderForIncrementalLocalState(train_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
        test_dataset = GPTDataloaderForIncrementalLocalState(test_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
        probing_dataset_train = IncrementalLocalStateProbeDataLoader(act_container_train, train_dataset)
        probing_dataset_test = IncrementalLocalStateProbeDataLoader(act_container_test, test_dataset)

    else:
        probing_dataset_train = ProbeDataLoader(act_container_train, dataset_path_train, object_map)
        probing_dataset_test = ProbeDataLoader(act_container_test, dataset_path_test, object_map)

    
    input_dim = probing_dataset_train[0][0].shape[-1]
 
    train_dataset, test_dataset = probing_dataset_train, probing_dataset_test 
    sampler = None


    if args.object_location:
        if args.twolayer:
            raise ValueError("Parameter --twolayer is not supported when using the object location probe.")
        probe = ObjectLocationProbeClassification(device,
            input_dim=input_dim,
            probe_class=probe_class,
            ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32))
    else:
        if args.twolayer:
            probe = BatteryProbeClassificationTwoLayer(device,
                input_dim=input_dim,
                probe_class=probe_class,
                num_task=100,
                mid_dim=args.mid_dim,
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                )             
        else: 
            probe = BatteryProbeClassification(device,
                input_dim=input_dim,
                probe_class=probe_class,
                num_task=100,
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                )        
        # num_task=100: the size of item vocab

    max_epochs = args.epo
    lr = args.learning_rate
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=max_epochs, batch_size=1024, learning_rate=lr,
        betas=(.9, .999), 
        lr_decay=True, warmup_tokens=len(train_dataset)*5, 
        final_tokens=len(train_dataset)*max_epochs,
        num_workers=4, weight_decay=0., 
        ckpt_path=os.path.join(args.checkpoint_root, folder_name, f"layer{args.layer}_token1")
    )
    # make sure ckpt_path exists
    if not os.path.exists(tconf.ckpt_path): # not sure why when using the incremental setting, it throws path notexists error. So add this line
        os.makedirs(tconf.ckpt_path)
    
    # pringing config
    print(f"Learning rate: {tconf.learning_rate}")
    print(f"Max epochs: {tconf.max_epochs}")
    print(f"Batch size: {tconf.batch_size}")
    
    
    if args.exp_name == "mentioned":
        trainer = Mention_Trainer(probe, train_dataset, test_dataset, tconf)
    else:
        # NOTE for incremental exp, should fit well with the standard trainer
        trainer = Trainer(probe, train_dataset, test_dataset, tconf)
    if not args.eval_only:
        predictions_matrix = trainer.train(prt=True).astype(int)
        trainer.save_traces()
        trainer.save_checkpoint()
    else:
        trainer.load_checkpoint()
        predictions_matrix = trainer.predict(prt=True).astype(int)
    
    predictions_file = os.path.join(tconf.ckpt_path, "predictions.txt")
    header = " ".join(object_list)
    np.savetxt(predictions_file, predictions_matrix, delimiter=" ", fmt='%i', header=header, comments="")
    
    if args.binary_probe:
    # save plot
        fig_file = os.path.join(tconf.ckpt_path, "predictions.pdf")
        trainer.flush_plot(fig_file)

        
        cm_file = os.path.join(tconf.ckpt_path, "confusion_matrix.txt")
        trainer.generate_confusion_matrix(cm_file)
        

if __name__ == "__main__":
    main()
    

