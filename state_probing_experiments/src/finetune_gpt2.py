# https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh
import argparse
import csv
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, Adafactor

from rich.table import Column, Table
from rich import box
from rich.console import Console


_MAX_LENGTH = 512
_MAX_NEW_TOKENS = 50


class LMDataloader(Dataset):
    """Loads LM training dataset with masked input."""

    def __init__(self, dataframe, tokenizer, max_length=_MAX_LENGTH):

        self.data = dataframe
        self.tokenizer = tokenizer
        self.input_text = self.data["sentence"]
        self.prefix_text = self.data["prefix"]
        self.max_length = max_length

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])

        inp = self.tokenizer.batch_encode_plus(
            [input_text], max_length=self.max_length, pad_to_max_length=True,
            padding="max_length", return_tensors='pt')
        
        prefix_text = str(self.prefix_text[index])
        pref = self.tokenizer.batch_encode_plus(
            [prefix_text], padding="do_not_pad", return_tensors='pt')
        pref_lens = [len(text) for text in pref["input_ids"]]
        
        input_ids = inp['input_ids'].squeeze()
        attn_masks = inp['attention_mask'].squeeze()

        return {
            'input_ids': input_ids.to(dtype=torch.long),
            'attn_masks': attn_masks.to(dtype=torch.long),
            'prefix_lens': torch.tensor(pref_lens, dtype=torch.long),
        }

    
class LMDataloaderForInference(Dataset):
    """Loads LM dataset for inference."""

    def __init__(self, dataframe, tokenizer, max_length=_MAX_LENGTH):

        self.data = dataframe
        self.tokenizer = tokenizer
        self.prefix_text = self.data["prefix"]
        self.target_text = self.data["sentence"]
        self.max_length = max_length

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        self.tokenizer.padding_side = "right"
        target_text = str(self.target_text[index])

        targ = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.max_length, pad_to_max_length=True,
            padding="max_length", return_tensors='pt')

        self.tokenizer.padding_side = "left"
        prefix_text = str(self.prefix_text[index])

        pref = self.tokenizer.batch_encode_plus(
            [prefix_text], max_length=self.max_length, pad_to_max_length=True,
            padding="max_length", return_tensors='pt')

        target_ids = targ['input_ids'].squeeze()
        prefix_ids = pref['input_ids'].squeeze()
        prefix_attn_masks = pref['attention_mask'].squeeze()

        return {
            'target_ids': target_ids.to(dtype=torch.long),
            'prefix_ids': prefix_ids.to(dtype=torch.long),
            'prefix_attn_masks': prefix_attn_masks.to(dtype=torch.long),
        }


def display_df(df):
    """Displays dataframe in ASCII format."""

    table = Table(Column("source_text", justify="center"), Column(
        "target_text", justify="center"), title="Sample Data", pad_edge=False, box=box.ASCII)

    for _, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)


def train(model, device, tokenizer,
          train_loader, train_epochs,
          optimizer, output_dir, save_every_n_epochs,
          ignore_prefix_loss):

    model.train()
    last_checkpoint_path = os.path.join(output_dir, 'last.ckpt')
    last_checkpoint_epoch = -1
    if os.path.exists(last_checkpoint_path):
        last_checkpoint = torch.load(last_checkpoint_path)
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        last_checkpoint_epoch = last_checkpoint['epoch']
        console.print(f"""[Model] Checkpoint information found. Resuming training from Epoch {last_checkpoint_path}\n""")

    for epoch in range(train_epochs):
        if epoch <= last_checkpoint_epoch:
            continue
        for step, data in enumerate(train_loader):
            labels = data['input_ids'].to(device, dtype=torch.long)
            labels[labels == tokenizer.pad_token_id] = -100

            if ignore_prefix_loss:
                for lab, l in zip(labels, data['prefix_lens']):
                    pref_mask = torch.arange(0, len(lab))
                    lab[pref_mask < l] = -100
                    
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attn_masks'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)

            loss = outputs[0]
            writer.add_scalar("Training loss", loss, step)

            if step % 100 == 0:
                training_logger.add_row(str(epoch), str(step), str(loss))
                console.print(f"Epoch {epoch}, Step: {step}, Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        resume_info = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(output_dir, exist_ok=True)
        torch.save(resume_info, os.path.join(output_dir, 'last.ckpt'))

        if save_every_n_epochs > 0:
            if epoch % save_every_n_epochs == 0:
                path = os.path.join(output_dir, f"model_files_ep{epoch}")
                log_path = os.path.join(output_dir, f"ep-{epoch}.log")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                console.save_text(log_path)
                console.print(f"""[Model] Model saved @ {path}\n""")


def predict(model, device, tokenizer, loader, split, output_dir, beam_size):
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    target_outputs = []
    orig_inputs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            targets = data['target_ids'].to(device, dtype = torch.long)
            ids = data['prefix_ids'].to(device, dtype = torch.long)
            mask = data['prefix_attn_masks'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids=ids,
              attention_mask=mask,
              max_new_tokens=_MAX_NEW_TOKENS,
              num_beams=beam_size,
#              repetition_penalty=2.5,
#              length_penalty=1.0,
              early_stopping=True
            )
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            targs = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in targets]
            inputs = [tokenizer.decode(
                inp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for inp in ids]

            if i % 10==0:
                console.print(f'Completed {i}\n')
                assert len(preds) == len(targs) == len(inputs)
                console.print('target\tpredicted\tinput\n')
                for pred, target, inp in zip(preds, targs, inputs):
                    console.print(f'{target}\t{pred}\t{inp}\n')
            predictions.extend(preds)
            target_outputs.extend(targs)
            orig_inputs.extend(inputs)

            with open(os.path.join(output_dir, f'predictions_{split}.tsv'), 'a') as wf:
                writer = csv.DictWriter(wf, delimiter='\t', fieldnames=['target', 'prediction', 'input'])
                if i == 0:
                    writer.writeheader()
                for pred, target, inp in zip(preds, targs, inputs):
                    pred_w = pred.removeprefix(inp + " ")
                    targ_w = target.removeprefix(inp + " ").strip(".")
                    inp_w = inp + ' .'
                    writer.writerow({'target': targ_w, 'prediction': pred_w, 'input': inp_w})

    return predictions, target_outputs, orig_inputs, accuracy_score(target_outputs, predictions)


def GPTTrainer(train_df, dev_df, test_df,
               output_dir,
               model,
               tokenizer,
               tokenizer_inference,
               train_batch_size,
               valid_batch_size,
               train_epochs,
               learning_rate,
               max_length,
               save_every=None,
               ignore_prefix_loss=False):
    """
    GPT trainer

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f'Device: {device}')
    model = model.to(device)

    console.log("[Data]: Reading data...\n")

    # train_df = train_df[[source_field, target_field]]
    # dev_df = dev_df[[source_field, target_field]]
    display_df(train_df.head(2))
    display_df(dev_df.head(2))

    console.print(f"TRAIN Dataset: {train_df.shape}")
    console.print(f"DEV Dataset: {dev_df.shape}\n")

    if test_df is not None:
        # test_df = test_df[[source_field, target_field]]
        display_df(test_df.head(2))
        console.print(f"TEST Dataset: {test_df.shape}\n")

    train_dataset = LMDataloader(train_df, tokenizer, max_length)
    dev_dataset = LMDataloader(dev_df, tokenizer, max_length)

    training_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_dataset, valid_batch_size, shuffle=False, num_workers=0)

    if test_df is not None:
        test_dataset = LMDataloaderForInference(test_df, tokenizer_inference, max_length)
        test_loader = DataLoader(test_dataset, valid_batch_size, shuffle=False, num_workers=0)

    optimizer = Adafactor(params=model.parameters(), lr=learning_rate, relative_step=False)

    # Training loop
    console.log('[Initiating finetuning]...\n')

    train(model, device, tokenizer, training_loader, train_epochs, optimizer,
          output_dir, save_every_n_epochs=save_every, ignore_prefix_loss=ignore_prefix_loss)

    console.log(f'[Finished finetuning after {train_epochs} epochs.]')

    if train_epochs > 0:
        save_path = os.path.join(output_dir, "model_files")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")

        console.save_text(os.path.join(output_dir,'training_logs.txt'))
        console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'training_logs.txt')}\n""")

    console.log("[Predicting with final checkpoint]...\n")

    loaders = {
        'train': training_loader,
        'dev': val_loader,
        'test': test_loader,
    }

    # eval_splits = ['dev'] if test_df is None else ['dev', 'test']
    eval_splits = ['test']

    os.makedirs(output_dir, exist_ok=True)
    for split in eval_splits:
        console.log(f"[Generating predictions on {split}...]\n")
        predictions, targets, orig_inputs, accuracy = predict(
            model, device, tokenizer, loaders[split], split, output_dir, args.beam_size
        )
        # final_df = pd.DataFrame({'target': targets, 'prediction': predictions,'input': orig_inputs})
        # final_df.to_csv(os.path.join(output_dir, f'predictions_{split}.tsv'), sep='\t')
        console.print(f"""[Prediction accuracy] {accuracy}""")
        console.print(f"""Prediction data saved @ {os.path.join(output_dir)}\n""")

    console.log("[Prediction Completed.]\n")


def main(args):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load datasets from path
    if args.condensed:
        dataset_path = os.path.join(args.dataset_path, '{}-gpt-condensed.tsv')        
    else:
        dataset_path = os.path.join(args.dataset_path, '{}-gpt.tsv')
    dataframes = {'train_df': None, 'dev_df': None, 'test_df': None}

    for split_df in dataframes.keys():
        split = split_df.split('_')[0]
        with open(dataset_path.format(split)) as textfile:
            dataframes[split_df] = pd.read_csv(textfile, delimiter='\t')
                
    console.print(dataframes['train_df'].sample(10))
    console.print(dataframes['dev_df'].sample(10))
    console.print(dataframes['test_df'].sample(10))

    # Get model parameters
    if args.model_name_or_checkpoint.startswith('gpt'):
        model_name = args.model_name_or_checkpoint
    else:
        model_name = f'{args.model_name_or_checkpoint}/model_files'
    console.log(f"""[Model]: Loading {model_name}...\n""")

    if args.random_init:
        console.log(f'Using randomly initialized {model_name}.')
        config = AutoConfig.from_pretrained(model_name)
        model = GPT2LMHeadModel(config=config)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer_left = AutoTokenizer.from_pretrained(model_name)
    tokenizer_left.pad_token = tokenizer.eos_token
    tokenizer_left.pad_token_id = tokenizer.eos_token_id
    tokenizer_left.padding_side = "left"

    model_params = {
        'model': model,
        'tokenizer': tokenizer,
        'tokenizer_inference': tokenizer_left,
        'train_batch_size': args.train_batch_size,
        'valid_batch_size': args.val_batch_size,
        'train_epochs': args.train_epochs,
        'learning_rate': args.learning_rate,
        'max_length': _MAX_LENGTH,
        'save_every': args.save_every,
        'ignore_prefix_loss': args.ignore_prefix_loss,
    }

    GPTTrainer(**dataframes,
               output_dir=args.output_path,
               **model_params)


if __name__ == '__main__':
    console = Console(record=True)
    writer = SummaryWriter()
    training_logger = Table(Column("Epoch", justify="center" ),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status",pad_edge=False, box=box.ASCII)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_checkpoint', default=None, type=str, required=True,
                        help='Name of model to use (e.g., "t5-base") or a path that contains the model checkpoint.')
    parser.add_argument('--prompt', default=None, type=str)
    parser.add_argument('--dataset_path', type=str, required=True,
                            help='Path to a directory that contains files of the form {split}-t5.jsonl')
    parser.add_argument('--test_disjoint', action='store_true', help='If set, we will use test-disjoint-vocab instead of test for eval.')
    parser.add_argument('--output_path', default=None, type=str, required=True)
    parser.add_argument('--seed', default=None, type=int, required=True)
    parser.add_argument('--train_epochs', default=100, type=int, required=False)
    parser.add_argument('--early_stopping', default=False, action='store_true')
    parser.add_argument('--save_every', default=0, type=int, help='Save every n epochs.')
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--random_init', action='store_true', help='If set, we will use a randomly initialized model.') 
    parser.add_argument('--ignore_prefix_loss', action='store_true', help='If set, loss will only be computed for the target answer part of the input.')
    parser.add_argument('--condensed', action='store_true', help='If set, we will use the version of the dataset that predicts all box states in one example.')

    args = parser.parse_args()
    console.print(args)
    main(args)
