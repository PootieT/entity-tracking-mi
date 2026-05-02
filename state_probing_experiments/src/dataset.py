import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import json
import pdb
from torch.utils.data import Dataset, DataLoader
from src.state_evals import generate_state_matrix, detect_removals, detect_local_removals

_GPT_MAX_LENGTH = 512
NUM_BOXES = 7

PROMPT = """Given the description after "Description:", write a true statement about all boxes and their contents according to the description after "Statement:".

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.
Statement: Box 3 contains the paper and the string.

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 2 contains the bag and the machine and the map.

Description: """
ALTFORM_PROMPT =  """Given the description after "Description:", write a true statement about all boxes and their contents according to the description after "Statement:". If a box is empty, write "Box X contains nothing".

Description: The car is in Box 0, the cross is in Box 1, the bag and the machine are in Box 2, the paper and the string are in Box 3, the bill is in Box 4, the apple and the cash and the glass are in Box 5, the bottle and the map are in Box 6.
Statement: Box 3 contains the paper and the string.

Description: The car is in Box 0, the cross is in Box 1, the bag and the machine are in Box 2, the paper and the string are in Box 3, the bill is in Box 4, the apple and the cash and the glass are in Box 5, the bottle and the map are in Box 6. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map in Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 2 contains the bag and the machine and the map.

Description: """

class LMDataloader(Dataset):
    """Loads LM (T5 denoising) dataset with masked input."""

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_field, target_field, include_empty=True, min_prev_objects=-1):
        self.tokenizer = tokenizer
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        if self.include_empty and self.min_prev_objects < 1:
            self.data = dataframe
        else:
            # filter all examples with empty boxes if include_empty is set to False
            f = dataframe[target_field].str.contains("nothing") | dataframe[target_field].str.contains("is empty")
            self.data = dataframe[-f]
            if self.min_prev_objects > 0:
                f = dataframe[target_field].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                self.data = self.data[f]
            self.data = self.data.reset_index()

        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_field]
        self.target_text = self.data[target_field]
            

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Cleaning data so as to ensure data is in string type
        source_text = source_text.split()
        target_text = target_text.split()
        

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True,
            is_split_into_words=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_len, pad_to_max_length=True,
            is_split_into_words=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class GPTDataloaderForInference(Dataset):
    """Loads LM dataset for inference."""

    def __init__(self, dataframe, tokenizer, max_length=_GPT_MAX_LENGTH, include_empty=True, condition_on="number", min_prev_objects=-1, include_prompt=False, is_altform=False):

        self.tokenizer = tokenizer
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.condition_on = condition_on
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.include_empty:
            self.data = dataframe
        else:
            # filter all examples with empty boxes if include_empty is set to False
            f = dataframe["masked_content"].str.contains("nothing") | dataframe["masked_content"].str.contains("is empty")
            self.data = dataframe[-f]
        
            if self.min_prev_objects > 0:
                f = dataframe["masked_content"].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                self.data = self.data[f]

            self.data = self.data.reset_index()

        # for the original t5 dataset, prefix ends with box number, while for the few-shot dataset, it ends with "contains"
        if any(self.data["prefix"].str.endswith("contains")):
            # a patchy solution to remove the contains from the prefix
            self.data["prefix"] = self.data["prefix"].apply(lambda x: x[:-9] if x.endswith("contains") else x)
            self.data['masked_content'] = self.data['masked_content'].apply(lambda x: x.replace("contains ", "").replace("<extra_id_0> ", ""))
        
        
        self.prefix_text = self.data["prefix"] 
        self.target_text = self.data["sentence"]
        self.max_length = max_length

        if self.min_prev_objects > 0:
            self.prefix_text = self.data.apply(lambda x: x["prefix"] + " " + " and ".join(x["masked_content"].split(" and ")[0:self.min_prev_objects]) + " and the", axis=1)

        elif self.condition_on == "contains":
            # add " contains" to prefix
            self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains")
        elif self.condition_on == "the":
            # add " contains the" to prefix
            self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains the")

        if include_prompt:
            if not is_altform:
                self.prefix_text = self.prefix_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
                self.target_text = self.target_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            else:
                self.prefix_text = self.prefix_text.apply(lambda x: ALTFORM_PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
                self.target_text = self.target_text.apply(lambda x: ALTFORM_PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])

            print(self.prefix_text[0])
            print(self.prefix_text[1])
            print("---------")
            print(self.target_text[1])
            

    def __len__(self):
        return len(self.target_text)
    
    
    def get_max_length(self):
        # get the max length of the tokenized dataset, padding to the model's max length would be expensive
        max_len = 0
        for data in self.target_text:
            targ = self.tokenizer.batch_encode_plus(
                [data], return_tensors='pt')
            seq_len = targ['input_ids'].squeeze().shape[0]
            
            if seq_len > max_len:
                max_len = seq_len
        return max_len

    def __getitem__(self, index):
        # self.tokenizer.padding_side = "right" # changed to left for batched inference
        self.tokenizer.padding_side = "left" 
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
            'target_ids': target_ids.to(dtype=torch.long),
            'prefix_ids': prefix_ids.to(dtype=torch.long),
            'prefix_attn_masks': prefix_attn_masks.to(dtype=torch.long),
        }


class GPTDataloaderForIncrementalLocalState(Dataset):
    """Loads LM dataset for inference of Incremental Local States."""

    def __init__(self, dataframe, tokenizer, max_length=_GPT_MAX_LENGTH, include_empty=True, condition_on="number", min_prev_objects=-1, include_prompt=False, object_map = {}):

        self.tokenizer = tokenizer
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.condition_on = condition_on
        self.n_objects = len(object_map)
        self.oti = object_map
        def get_numops_global(s):
            s = "Description" +  s.split("Description")[-1]
            seq = [st.strip() for st in s.split('.') if st]
            if len(seq) <= 1:
                return 0
            else:
                return len(seq) - 1
        if self.include_empty:
            self.data = dataframe
        else:
            # filter all examples with empty boxes if include_empty is set to False
            f = dataframe["masked_content"].str.contains("nothing") | dataframe["masked_content"].str.contains("is empty")
            self.data = dataframe[-f]
        
            if self.min_prev_objects > 0:
                f = dataframe["masked_content"].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                self.data = self.data[f]

            self.data = self.data.reset_index()

        # for the original t5 dataset, prefix ends with box number, while for the few-shot dataset, it ends with "contains"
        if any(self.data["prefix"].str.endswith("contains")):
            # a patchy solution to remove the contains from the prefix
            self.data["prefix"] = self.data["prefix"].apply(lambda x: x[:-9] if x.endswith("contains") else x)
            self.data['masked_content'] = self.data['masked_content'].apply(lambda x: x.replace("contains ", "").replace("<extra_id_0> ", ""))
        
        
        self.prefix_text = self.data["prefix"] 
        self.target_text = self.data["sentence"]
        self.max_length = max_length

        if self.min_prev_objects > 0:
            self.prefix_text = self.data.apply(lambda x: x["prefix"] + " " + " and ".join(x["masked_content"].split(" and ")[0:self.min_prev_objects]) + " and the", axis=1)

        elif self.condition_on == "contains":
            # add " contains" to prefix
            self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains")
        elif self.condition_on == "the":
            # add " contains the" to prefix
            self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains the")
            
        # Already ends with the box id when using this dataloader. Need to remove the query completely
        

        if include_prompt:
            # self.prefix_text = self.prefix_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".")
            self.target_text = self.target_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.iloc[::NUM_BOXES].reset_index(drop=True)
            self.target_text = self.target_text.iloc[::NUM_BOXES].reset_index(drop=True)
            mask = self.prefix_text.apply(get_numops_global) > 0
            self.prefix_text = self.prefix_text[mask].reset_index(drop=True)
            self.target_text = self.target_text[mask].reset_index(drop=True)
            # also drop all the zero-ops samples with no operations
            

            print(self.prefix_text[0])
            print(self.prefix_text[1])
            print("---------")
            print(self.target_text[1])
        else:
            self.prefix_text = self.prefix_text.apply(lambda x:  ". ".join(x.split(". ")[:-1]) + ".")
            self.target_text = self.target_text.apply(lambda x:  ". ".join(x.split(". ")[:-1]) + ". " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.iloc[::NUM_BOXES].reset_index(drop=True)
            self.target_text = self.target_text.iloc[::NUM_BOXES].reset_index(drop=True)
            mask = self.prefix_text.apply(get_numops_global) > 0
            self.prefix_text = self.prefix_text[mask].reset_index(drop=True)
            self.target_text = self.target_text[mask].reset_index(drop=True)
            
        
            

            

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        self.tokenizer.padding_side = "right"
        target_text = str(self.target_text[index])

        targ = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.max_length, return_tensors='pt')

        prefix_text = str(self.prefix_text[index])

        pref = self.tokenizer.batch_encode_plus(
            [prefix_text], max_length=self.max_length, return_tensors='pt')

        target_ids = targ['input_ids'].squeeze()
        prefix_ids = pref['input_ids'].squeeze()
        prefix_attn_masks = pref['attention_mask'].squeeze()
        # only keep the last example. Remove the few shot examples.
        if "Description" in prefix_text:
            clean_prefix = "Description" +  prefix_text.split("Description")[-1] # This is used to generate the state matrix, so it should not affect the tokenization or saving activations with the LMs.
        else:
            clean_prefix = prefix_text
            
        start_of_task = None # Find the start of the task description in the tokenized sequence
        
        
        # Generate Mentioned Vectors from clean_prefix
        mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
        # pdb.set_trace()
        s_parts = clean_prefix.strip(".").split(".")
        o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") # this is a regex to find all object names in the sentence
        for o in o_names: 
            oidx = self.oti[o]
            mentioned_objects[oidx] = 1


        state_matrix, _, _ = generate_state_matrix(clean_prefix, object_map=self.oti, contains_query=False)
        state_matrix = torch.tensor(state_matrix, dtype=torch.float32)
        if "Description" in prefix_text:
            
            start_of_task = [i for i, tkn in enumerate(self.tokenizer.convert_ids_to_tokens(prefix_ids)) if 'Description' in tkn][-1]
        else:
            start_of_task = 0
        
        # 1. Find all the box ids after the start_of_task
        # 2. parse them according to the operations 
        # 3. remove the inital state -- at that time BOX_ID position does not see any contents yet.
        # 4. We shuold have something like [[box ids in op1], [box ids in op2], ...], outer list length = num_ops, and a list of state matrix like [[content of boxes in op1], [content of boxes in op2], ...], outer list length = num_ops, inner list length = length of box ids in that operation
        seq = [st.strip() for st in clean_prefix.split('.') if st]
        ori_state = seq[0]
        op_seq = seq[1:] # Might be a problem -- no query now
        box_ids_in_ops = []
        for op in op_seq:
            box_ids = re.findall(r'Box (\d+)', op)
            if len(box_ids) > 0:
                box_ids_in_ops.append([int(bid) for bid in box_ids])
            else:
                box_ids_in_ops.append([])
        box_states_in_ops = []
        # if there's no operation at all, we should not remove the initial state
        for t in range(1, state_matrix.shape[0]): # would accidentally remove the final state when there's no operation at all
            box_states_in_ops.append([state_matrix[t, bid, :] for bid in box_ids_in_ops[t-1]])
        # now handle the last query box: add the id to the box_ids_in_ops, and add the final state to the box_states_in_ops. This will eliminate the need to process ops=0 case separately
        
        

        # now we convert the box ids to box position in the original prompt (used for saving activations)
        box_positions_in_ops = []
        flattened_box_ids = []
        # first flatten the box ids
        for bids in box_ids_in_ops:
            flattened_box_ids.extend(bids)

        # find all the id position in the tokenized prefix
        # start of task is not enough, need to also skip the initial state description
        # manually skip the first 7 box ids, a very hacky solution, but should work for now.
        box_id_positions = []
        num_id_encountered = 0
        for i, tkn in enumerate(self.tokenizer.convert_ids_to_tokens(prefix_ids)):
            # just find all the box ids (integers)
            # if is interger and i > start_of_task, need to check token prefix
            clean_token = tkn.replace("Ġ", "").replace("▁", "") # for different tokenizers
            if re.match(r'^\d+$', clean_token) and i > start_of_task:
                num_id_encountered += 1
                if num_id_encountered > NUM_BOXES: # skip the first 7 box ids, which should correspond to the initial state description
                    box_id_positions.append(i)
                    
        # box_id_positions = [i for i, tkn in enumerate(self.tokenizer.convert_ids_to_tokens(prefix_ids)) if re.match(r'Box', tkn) and i > start_of_task]
        assert len(box_id_positions) == len(flattened_box_ids), f"Number of box ids {len(flattened_box_ids)} does not match number of box id positions {len(box_id_positions)}!"
        # reformat into the original two-level structure
        idx = 0
        for bids in box_ids_in_ops:
            box_positions_in_ops.append(box_id_positions[idx:idx+len(bids)])
            idx += len(bids)
        assert idx == len(box_id_positions), f"Index {idx} does not match number of box id positions {len(box_id_positions)}!"
        
        # should use the flattened version of box box id position and box local states -- easier for indexing in the model activations
        box_positions_flattened = []
        box_states_flattened = []
        for bpos, bstate in zip(box_positions_in_ops, box_states_in_ops):
            box_positions_flattened.extend(bpos)
            box_states_flattened.extend(bstate)
        
            
        
        
                
        
        return {
            'target_ids': target_ids.to(dtype=torch.long),
            'prefix_ids': prefix_ids.to(dtype=torch.long),
            'prefix_attn_masks': prefix_attn_masks.to(dtype=torch.long),
            'box_id_positions': box_positions_in_ops, 
            'box_local_states': box_states_in_ops,
            'box_id_positions_flattened': box_positions_flattened,
            'box_local_states_flattened': torch.stack(box_states_flattened),
            'mentioned_objects': mentioned_objects
        }
        
    def build_dummy_activations(self, hidden_dim=5120):
        """Build dummy activations for testing, according to box_id_positions_flattened length."""
        dummy_activations = []
        for i in range(len(self)):
            data = self[i]
            n_box_ids = len(data['box_id_positions_flattened'])
            dummy_activations.append(torch.zeros((n_box_ids, hidden_dim), dtype=torch.float32))
        return dummy_activations
    

class IncrementalLocalStateProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing. This class mostly serves as a unwrapper for the GPTDataloaderForInference class -- for each example, the activations are saved one tensor. But we need to unroll them according to the box ids and local states."""
    def __init__(self, activations, dataset):
        self.activations = activations # shape: N_Prompts x [boxids x hidden_dim]
        self.dataset = dataset # shape: (N_Prompts x N_Boxes)
        self.oti = dataset.oti
        # pdb.set_trace()
        self.expanded_activations, self.labels, self.num_ops, self.counts, self.all_mentioned_objects = self.expand_examples(dataset, None)


    def expand_examples(self, dataset, path_to_data):
        """
        dataset: the GPTDataloaderForInference dataset
        path_to_data: path to the original data file, used to compute the mentioned vectors (non-trivial cases, but probably we can just focus on the recall rate (acc for positive examples) for now.)
        
        """
        expanded_activations = []
        labels = []
        num_ops = []
        counts = np.zeros(2)
        all_mentioned_objects = []
        # mentioned_objects = [] # ignored for now
        
        for i in range(len(dataset)):
            # pdb.set_trace(header="debug expand examples")
            data = dataset[i]
            act = self.activations[i] # shape: [N_box_ids, hidden_dim]
            print(i, act.shape, len(data['box_id_positions_flattened']), data['box_local_states_flattened'].shape)
            assert act.shape[0] == len(data['box_id_positions_flattened']) == data['box_local_states_flattened'].shape[0], f"Activation shape {act.shape[0]} does not match number of box ids {len(data['box_id_positions_flattened'])} or number of box states {data['box_local_states_flattened'].shape[0]}!"
            num_boxes_to_unroll = len(data['box_id_positions_flattened'])
            for i in range(num_boxes_to_unroll):
                box_id = data['box_id_positions_flattened'][i]
                box_state = data['box_local_states_flattened'][i]
                expanded_activations.append(act[i])
                labels.append(box_state)
                
                num_ops.append([len(data['box_id_positions']) - 1] * len(self.oti)) # all objects share the same number of operations
                all_mentioned_objects.append(data['mentioned_objects'])
                counts += np.array([torch.sum((box_state == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)

        return expanded_activations, labels, num_ops, counts, all_mentioned_objects

    def __getitem__(self, index):
        return self.expanded_activations[index], self.labels[index], torch.tensor(self.num_ops[index]).to(torch.long), self.all_mentioned_objects[index]
    def __len__(self):
        return len(self.expanded_activations)
    
    def get_weights(self):
        weights = torch.tensor([np.sum(self.counts)], dtype=torch.float32) / torch.tensor(self.counts * (NUM_BOXES + 1), dtype=torch.float32) # why nan?
        return weights
    


class ProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, object_to_index_map):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices. 
        """
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        self.examples, self.num_ops, counts, self.mentioned_objects = self.load_examples(path_to_data)
        
        # Activate this for debugging on a handful of examples
        #self.examples = self.examples[:10]
        #self.num_ops = self.num_ops[:10]
        
        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        
        assert len(self.activations) == NUM_BOXES * len(self.examples)
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], self.examples[index // NUM_BOXES], torch.tensor(self.num_ops[index // NUM_BOXES]).to(torch.long),  self.mentioned_objects[index // NUM_BOXES]
    
    def load_examples(self, path_to_data):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
        
        
        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"
        
        counts = np.zeros((NUM_BOXES + 1))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        for i, ex in enumerate(raw_examples):
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            box_no = int(s[4]) # 4th character is the box number
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    oidx = self.oti[c]
                    box_contents[oidx] = box_no + 1
            
            if (i % NUM_BOXES) == (NUM_BOXES - 1):
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(NUM_BOXES + 1)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                box_contents = torch.zeros(self.n_objects)
                
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                # o_names = re.findall(r'the ([^ ,.]+) ', " ".join(s_parts[:-1]) + " ")
                o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") # this is a regex to find all object names in the sentence
                for o in o_names: 
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1
                all_mentioned_objects.append(mentioned_objects)

                
        
        return examples, num_ops, counts, all_mentioned_objects
    


    


class BinaryProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, object_to_index_map, include_empty=True, min_prev_objects=-1):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices. 
        """
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects # what is this.
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        # pdb.set_trace(header="debug loader")
        self.examples, self.num_ops, counts, self.mentioned_objects = self.load_examples(path_to_data)
        
        # Activate this for debugging on a handful of examples
        #self.examples = self.examples[:70]
        #self.num_ops = self.num_ops[:70]
        
        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        
        assert len(self.activations) ==  len(self.examples)
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index]
    
    def load_examples(self, path_to_data):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
        
        
        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"
        
        counts = np.zeros((2))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        for i, ex in enumerate(raw_examples): # TODO understand whats happening here
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            is_empty = True
            n_obj = 0
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    # contents looks like a list of strings, i.e. the objects in the box
                    # we need to compute 1. how many objects are in this box
                    n_obj += 1
                    # only consider objects that haven't been output already
                    if self.min_prev_objects < 1 or n_obj > self.min_prev_objects:
                        # add new object to box_contents
                        oidx = self.oti[c]
                        box_contents[oidx] = 1

            
            if (not is_empty or self.include_empty) and n_obj > self.min_prev_objects:
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects) # TODO: this will overflow: 
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                # o_names = re.findall(r'the ([^ ,.]+) ', " ".join(s_parts[:-1]) + " ")
                o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") 
                for o in o_names: 
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1 # store a binary vector of mentioned objects
                    
                # need to assert box contents are all in the mentioned objects
                # if torch.sum(box_contents * mentioned_objects) == torch.sum(box_contents):
                #     print("warning")
                # TODO this assertion is not true because of the regular expression used to extract object names, need to fix it ASAP.
                box_contents = torch.zeros(self.n_objects)
                all_mentioned_objects.append(mentioned_objects)
                
        # log basic stats of the distributions
        print("Class distribution:", counts)
        print("Number of examples:", len(examples))
        print("Number of Non-Trivial examples:", np.sum(all_mentioned_objects)) # number of mentioned objects
        print("Number of positve samples", counts[1])
        print("Number of negative samples", counts[0])
        
        return examples, num_ops, counts, all_mentioned_objects








class ObjectLocationProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices. 
        """
        self.examples, self.num_ops, counts = self.load_examples(path_to_data)
        
        # Activate this for debugging on a handful of examples
        #self.examples = self.examples[:500]
        #self.num_ops = self.num_ops[:500]
        
        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)        

        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]

        
        assert len(self.activations) == len(self.examples)
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], torch.tensor([self.examples[index]]), torch.tensor([self.num_ops[index]]).to(torch.long)
    
    def load_examples(self, path_to_data):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
                
        y = []
        num_ops = []
        counts = np.zeros(NUM_BOXES + 1)
        for ex in raw_examples:
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            if "no box" not in ex["masked_content"]: # why no box?
                # box_no = int(s[-1]) # 4th character is the box number
                box_no = int(s[4])
                y.append(box_no + 1)
                counts[box_no + 1] += 1
            else:
                y.append(0)
                counts[0] += 1

            num_ops.append(len(s_parts) - 2)


        return y, num_ops, counts




class GlobalProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, object_to_index_map, include_empty=True, min_prev_objects=-1):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices. 
        """
        self.include_empty = include_empty
        assert include_empty == True, "Global Probing Experiments Requires complete dataset with Empty Boxes!"
        self.min_prev_objects = min_prev_objects
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        self.examples, self.global_states, counts = self.load_examples(path_to_data)
        # examples are of the same length as the raw examples, but context is of length of len(examples) // NUM_BOXES, since they share the same context(state + operation sequences)
        
        # Activate this for debugging on a handful of examples
        #self.examples = self.examples[:70]
        #self.num_ops = self.num_ops[:70]
        
        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        print(len(self.activations), len(self.examples))
        assert len(self.activations) ==  len(self.examples)
        
        self.n = len(self.global_states)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index * NUM_BOXES], self.global_states[index]

    def load_examples(self, path_to_data):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
        
        
        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"
        
        counts = np.zeros((NUM_BOXES + 1))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        global_states = [] 
        for i, ex in enumerate(raw_examples):
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            is_empty = True
            n_obj = 0
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    n_obj += 1
                    # only consider objects that haven't been output already
                    if self.min_prev_objects < 1 or n_obj > self.min_prev_objects:
                        oidx = self.oti[c]
                        box_contents[oidx] = 1

            if (not is_empty or self.include_empty) and n_obj > self.min_prev_objects:
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                box_contents = torch.zeros(self.n_objects)
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                # o_names = re.findall(r'the ([^ ,.]+) ', " ".join(s_parts[:-1]) + " ")
                o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") # this is a regex to find all object names in the sentence
                for o in o_names: 
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1
                all_mentioned_objects.append(mentioned_objects)
        counts = np.zeros((NUM_BOXES + 1))
        for i in range(len(examples)):
            # add context to examples
            if i % NUM_BOXES != 0:
                continue
            assignment = torch.zeros(self.n_objects, dtype=torch.long)
            for j in range(NUM_BOXES):
                box_contents = examples[i + j] # n_objects x 1
                box_no = j + 1
                counts[box_no] += torch.sum(box_contents).item()
                assignment[box_contents == 1] = box_no
            global_states.append(assignment)
            
                        
        return examples, global_states, counts



class MentionedProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, object_to_index_map, include_empty=True, min_prev_objects=-1):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices. 
        """
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        self.examples, self.num_ops, counts, self.mentioned_objects, self.removed_objects = self.load_examples(path_to_data)
        # THE ONLY DIFFRENRCE IT THAT COUNTS KEEP TRACK OF MENTIONED/NOT MENTIONED OBJECTS, NOT OBJECT STATES
        # Activate this for debugging on a handful of examples
        #self.examples = self.examples[:70]
        #self.num_ops = self.num_ops[:70]
        
        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        print(len(self.activations), len(self.examples))
        assert len(self.activations) ==  len(self.examples)
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index], self.removed_objects[index]
    
    def load_examples(self, path_to_data):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
        
        
        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"
        
        counts = np.zeros((2))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        all_removed_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        for i, ex in enumerate(raw_examples):
            s_parts = ex["sentence"].strip(".").split(".")
            state_mat, removed_objs, _ = generate_state_matrix(ex["sentence"], self.oti, num_boxes=NUM_BOXES, num_obj=self.n_objects)
            s = s_parts[-1].strip()
            is_empty = True
            n_obj = 0
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    n_obj += 1
                    # only consider objects that haven't been output already
                    if self.min_prev_objects < 1 or n_obj > self.min_prev_objects:
                        oidx = self.oti[c]
                        box_contents[oidx] = 1

            
            if (not is_empty or self.include_empty) and n_obj >= self.min_prev_objects: # just to include empty boxes
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                box_contents = torch.zeros(self.n_objects)
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                removed_objects = torch.zeros(self.n_objects) #vector with removed objects
                for oidx in removed_objs:
                    removed_objects[oidx] = 1
                # o_names = re.findall(r'the ([^ ,.]+) ', " ".join(s_parts[:-1]) + " ")
                o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") # this is a regex to find all object names in the sentence
                for o in o_names: 
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1
                all_mentioned_objects.append(mentioned_objects)
                all_removed_objects.append(removed_objects)
                

        # patchy solution, override counts with the sum of mentioned objects

        # all mentioned objects: list of tensors, each tensor is of shape (n_objects,), but every #NUM_BOXES-th tensor is the same, since they share the same context
        counts = np.zeros((2))
        counts[0] = torch.sum(torch.stack(all_mentioned_objects) == 0).item()
        counts[1] = torch.sum(torch.stack(all_mentioned_objects) == 1).item()
        # may need to divide by NUM BOXES but lets leave it for now
        
        print("Class distribution:", counts)
        print("Number of examples:", len(examples))
        print("Number of Non-Trivial examples:", np.sum(all_mentioned_objects)) # number of mentioned objects
        print("Number of positve samples", counts[1])
        print("Number of negative samples", counts[0])
        
        return examples, num_ops, counts, all_mentioned_objects, all_removed_objects







 


















