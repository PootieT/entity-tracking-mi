"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging

from tqdm import tqdm
import numpy as np
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

MAX_NUM_OPERATIONS = 13 # previously set to 12, but the max number of operations in the dataset is 13, so we set it to 13 

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            
        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []
        # would be a list of T-long, each is a lits of MAX_NUM_OPERATIONS-long, for stratified accuracies        
        self.train_strat_acc_cont = []
        self.test_strat_acc_cont = []
        self.training_history = [] # list of dict, each dict contains the training and testing metrics for each epoch, for easier analysis and plotting later
        self.test_history = [] # list of dict, each dict contains the testing metrics for each epoch, for easier analysis and plotting later
    def flush_plot(self, save_path=None):
        # plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
        axs = axs.flat
        axs[0].plot(self.train_loss_cont, label="train")
        axs[0].plot(self.test_loss_cont, label="test")
        axs[0].set_title("Loss")
        axs[0].legend()
        axs[1].plot(self.train_acc_cont, label="train")
        axs[1].plot(self.test_acc_cont, label="test")
        axs[1].set_title("Accuracy")
        axs[1].legend()
        if save_path is not None:
            fig.savefig(save_path)
        # plt.show()
        # return a figure object
        

    def save_traces(self, ):
        tbd = {
            "train_loss_cont": self.train_loss_cont, "test_loss_cont" :self.test_loss_cont, 
            "train_acc_cont": self.train_acc_cont, "test_acc_cont": self.test_acc_cont, 
            "train_strat_acc_cont": self.train_strat_acc_cont, "test_strat_acc_cont": self.test_strat_acc_cont, 
        }
        with open(os.path.join(self.config.ckpt_path, "tensorboard.txt"), "w") as f:
            f.write(json.dumps(tbd) + "\n")
            
        with open(os.path.join(self.config.ckpt_path, "training_history.jsonl"), "w") as f:
            for record in self.training_history:
                f.write(json.dumps(record) + "\n")
        with open(os.path.join(self.config.ckpt_path, "test_history.jsonl"), "w") as f:
            for record in self.test_history:
                f.write(json.dumps(record) + "\n")

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(raw_model.state_dict(), os.path.join(self.config.ckpt_path, "checkpoint.ckpt"))

    def load_checkpoint(self):
        torch.load( os.path.join(self.config.ckpt_path, "checkpoint.ckpt"), map_location=self.device)

    def predict(self, prt=True):
        model, config = self.model, self.config
        model.train(False)
        data = self.test_dataset
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        
        pbar = enumerate(loader)
        losses = []
        totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
        hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        predictions = []
        gt_ls = []
        mentioned_ls = []
        
        
        for _, (x, y, age, mentioned) in pbar:
            gt_ls.append(y.cpu().numpy())
            mentioned_ls.append(mentioned.cpu().numpy())
            
            x = x.to(self.device)  # [B, f]
            y = y.to(self.device)  # [B, #task=64] 
            age = age.to(self.device)  # [B, #task=64], in 0--59
            mentioned = mentioned.to(self.device) # [B, #task]
            with torch.set_grad_enabled(False):
                logits, loss = model(x, y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)

                y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                hits = y_hat == y  # [B, #task]
                hits_nontrivial = (y_hat == y) * mentioned
                hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                predictions.append(y_hat.cpu().numpy())
                
        test_loss = float(np.mean(losses))
        test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
        test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()

        if prt: 
            logger.info(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%;  test non-triv acc {test_acc_nontriv*100:.2f}%")
        predictions_matrix = np.concatenate(predictions, axis=0)
        
        # need a confusion matrix for all classes, as well as trivial/non-trivial cases
        # we have prediction matrix (binary or tenary, not convertable), ground truth y(binary or ternary, convertable) and mentioned matrix
        # two cases, if it's a binary probe, we expand the confusion matrix to 2x3, if it's ternary, we just use 3x3
        self.predictions = predictions_matrix
        self.ground_truth = np.concatenate(gt_ls, axis=0)
        self.mentioned = np.concatenate(mentioned_ls, axis=0)
        # self.generate_confusion_matrix(
        #     predictions_matrix,
        #     np.concatenate(gt_ls, axis=0),
        #     np.concatenate(mentioned_ls, axis=0)
        # )
        return predictions_matrix
    
    
    
   
    def train(self, prt=True):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, scheduler = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
            hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_trivial_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_triv = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            predictions = []
            recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FN)
            precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FP)
            total_recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FN 
            total_precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FP
            
            
            # Other metrics might be interesting -- False positive classes, are they non-trivial cases or just random?
            # FP - not mentioned -- just random noise probably
            # FP - mentioned -- confused with objects in other boxes, interesting.

            total_FP_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)
            total_TP_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  
            total_TN_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)
            total_FN_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)
            total_mentioned_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)
            
            total_TN_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)
            total_TN_triv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)
            FP_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # corresponding to false-positive in the non-trivial cases
            FP_triv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # corresponding to false-positive in the trivial cases
            
            
            
            gt_ls = []
            mentioned_ls = []
            pbar = tqdm(enumerate(loader), total=len(loader), disable=not prt) if is_train else enumerate(loader)
            for it, (x, y, age, mentioned) in pbar: # self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index]
                x = x.to(self.device)  # [B, f]
                y = y.to(self.device)  # [B, #task=64] 
                age = age.to(self.device)  # [B, #task=64], in 0--59
                mentioned = mentioned.to(self.device) # [B, #task]
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_triv += np.array([torch.sum((age == i) * (1 - mentioned)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                    hits = y_hat == y  # [B, #task]
                    hits_nontrivial = (y_hat == y) * mentioned
                    hits_trivial = hits * (1 - mentioned)  # hits that are trivial (not mentioned)
                    hits_trivial_epoch += np.array([torch.sum(hits_trivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range
                                            (MAX_NUM_OPERATIONS)]).astype(float)
                    hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    predictions.append(y_hat.cpu().numpy())
                    gt_ls.append(y.cpu().numpy())
                    mentioned_ls.append(mentioned.cpu().numpy())
                    
                    recalls_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    precision_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_recalls_epoch += np.array([torch.sum((y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_precision_epoch += np.array([torch.sum((y_hat == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    
                    total_FP_epoch += np.array([torch.sum((y_hat == 1) * (y == 0) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    FP_nontriv_epoch += np.array([torch.sum((y_hat == 1) * (y == 0) * mentioned * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    FP_triv_epoch += np.array([torch.sum((y_hat == 1) * (y == 0) * (1 - mentioned) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    
                    total_TP_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_TN_epoch += np.array([torch.sum((y_hat == 0) * (y == 0) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_FN_epoch += np.array([torch.sum((y_hat == 0) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_mentioned_epoch += np.array([torch.sum(mentioned * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_TN_nontriv_epoch += np.array([torch.sum((y_hat == 0) * (y == 0) * mentioned * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_TN_triv_epoch += np.array([torch.sum((y_hat == 0) * (y == 0) * (1 - mentioned) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    #print(hits_nontriv_epoch)
                    #print(totals_epoch_nontriv)


                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    mean_loss = float(np.mean(losses))
                    mean_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                    mean_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()
                    mean_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                    mean_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                    FP_nontriv_rate = np.sum(FP_nontriv_epoch).item() / np.sum(total_FP_epoch).item() if np.sum(total_FP_epoch).item() > 0 else 0.0
                    FP_triv_rate = np.sum(FP_triv_epoch).item() / np.sum(total_FP_epoch).item() if np.sum(total_FP_epoch).item() > 0 else 0.0


                    # Some stats: Total Samples, TP (true positive) , TN (true negative: not mentioned + elsewhere), FP (not mentioned + elsewherem, classified as positive) , FN (present, classified as negative),

                    # logger.info(f"STAT: Total {np.sum(totals_epoch).item()}, TP {np.sum(hits_nontriv_epoch).item()}, TN {np.sum(total_TN_epoch).item()}, FP {np.sum(FP_nontriv_epoch).item()}, FN {np.sum(total_FN_epoch).item()}, FP_triv {np.sum(FP_triv_epoch).item()}, FP_non_triv {np.sum(FP_nontriv_epoch).item()}, Total_mentioned {np.sum(total_mentioned_epoch).item()}, Total_TN_nontriv {np.sum(total_TN_nontriv_epoch).item()}, Total_TN_triv {np.sum(total_TN_triv_epoch).item()}")

                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(f"epoch {epoch+1}: train loss {mean_loss:.5f}; lr {lr:.2e}; train acc {mean_acc*100:.2f}%; train non-triv acc: {mean_acc_nontriv*100:.2f}; trivial acc: {np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item()*100:.2f}%; recall {mean_recall*100:.2f}%; precision {mean_precision*100:.2f}%; FP non-triv rate {FP_nontriv_rate*100:.2f}%; FP triv rate {FP_triv_rate*100:.2f}%")
                    self.training_history.append({
                        "epoch": epoch+1,
                        "train_loss": mean_loss,
                        "train_acc": mean_acc,
                        "train_nontriv_acc": mean_acc_nontriv,
                        "train_triv_acc": np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item() if np.sum(totals_epoch_triv).item() > 0 else 0.0,
                        "train_recall": mean_recall,
                        "train_precision": mean_precision,
                        "FP_nontriv_rate": FP_nontriv_rate,
                        "FP_triv_rate": FP_triv_rate,
                    })
            if is_train:
                self.train_loss_cont.append(mean_loss)
                self.train_acc_cont.append(mean_acc)
                self.train_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())

            if not is_train:
                test_loss = float(np.mean(losses))
                scheduler.step(test_loss)
                test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()
                test_acc_trivial = np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item()
                
                test_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                test_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                
                test_fp_nontriv_rate = np.sum(FP_nontriv_epoch).item() / np.sum(total_FP_epoch).item() if np.sum(total_FP_epoch).item() > 0 else 0.0
                test_fp_triv_rate = np.sum(FP_triv_epoch).item() / np.sum(total_FP_epoch).item() if np.sum(total_FP_epoch).item() > 0 else 0.0
                

                if prt: 
                    # also log tp, fp, fn for overall
                    logger.info(f"STAT: Total {np.sum(totals_epoch).item()}, TP {np.sum(hits_nontriv_epoch).item()}, TN {np.sum(total_TN_epoch).item()}, FP {np.sum(total_FP_epoch).item()}, FN {np.sum(total_FN_epoch).item()}, FP_triv {np.sum(FP_triv_epoch).item()}, FP_non_triv {np.sum(FP_nontriv_epoch).item()}, Total_mentioned {np.sum(total_mentioned_epoch).item()}, Total_TN_nontriv {np.sum(total_TN_nontriv_epoch).item()}, Total_TN_triv {np.sum(total_TN_triv_epoch).item()}")

                    logger.info(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%;  test non-triv acc {test_acc_nontriv*100:.2f}%;    test trivial acc {test_acc_trivial*100:.2f}%; recall {test_recall*100:.2f}%; precision {test_precision*100:.2f}%; FP non-triv rate {test_fp_nontriv_rate*100:.2f}%; FP triv rate {test_fp_triv_rate*100:.2f}%")
                self.test_history.append({
                    "epoch": epoch+1,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_nontriv_acc": test_acc_nontriv,
                    "test_triv_acc": test_acc_trivial,
                    "test_recall": test_recall,
                    "test_precision": test_precision,
                    "test_fp_nontriv_rate": test_fp_nontriv_rate,
                    "test_fp_triv_rate": test_fp_triv_rate,
                })
                self.test_loss_cont.append(test_loss)
                self.test_acc_cont.append(test_acc)
                self.test_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
                predictions_matrix = np.concatenate(predictions, axis=0)
                self.predictions = predictions_matrix
                self.ground_truth = np.concatenate(gt_ls, axis=0)
                self.mentioned = np.concatenate(mentioned_ls, axis=0)
                return test_loss, predictions_matrix

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss, predictions_matrix = run_epoch('test')
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()
                
        # return predictions after last epoch
        return predictions_matrix
    
    def generate_confusion_matrix(self, file_path=None):
        """
        Generate a confusion matrix for the predictions against the ground truth.
        If mentioned is provided, it will be used to filter out trivial cases.
        predictions: numpy array of shape [N, N_Obj] (binary or ternary)
        ground_truth: numpy array of shape [N, N_Obj] (binary or ternary)
        mentioned: numpy array of shape [N, N_Obj] (binary), optional
        """
        
        def print_confusion_matrix(cm, pred_labels, true_labels, file_path=file_path):
            with open(file_path, "w") as f:
                f.write("Confusion Matrix:\n")
                f.write(" " * 22 + " | " + " | ".join(f"{label:^20}" for label in true_labels) + " |\n")
                f.write("-" * (25 + len(true_labels) * 24) + "\n")
                for i, row in enumerate(cm):
                    row_str = " | ".join(f"{val:^20}" for val in row)
                    f.write(f"{pred_labels[i]:<22} | {row_str} |\n")
                f.write("-" * (25 + len(true_labels) * 24) + "\n")
                
        predictions = self.predictions
        ground_truth = self.ground_truth
        mentioned = self.mentioned 
        D, N = predictions.shape
        # logging.info(f"predictions shape: {predictions.shape}, ground_truth shape: {ground_truth.shape}, mentioned shape: {mentioned.shape}")
        assert D == ground_truth.shape[0] and N == ground_truth.shape[1], "Predictions and ground truth must have the same shape."
        assert D == mentioned.shape[0] and N == mentioned.shape[1], "Predictions and mentioned must have the same shape."
        
        assert np.all(np.isin(predictions, [0, 1, 2])), "Predictions must be binary or ternary (0, 1, or 2)."
        assert np.all(np.isin(ground_truth, [0, 1, 2])), "Ground truth must be binary or ternary (0, 1, or 2)."
        
        y_three_class = np.full_like(ground_truth, fill_value=0)  
        num_class = predictions.max() + 1  # 0 or 1 or 2
        y_three_class[mentioned == 1] = ground_truth[mentioned == 1] + 1 if num_class == 2 else ground_truth[mentioned == 1]# y ∈ {0,1} → {1,2}
        assert y_three_class.shape == ground_truth.shape, "y_three_class must have the same shape as ground_truth."
        assert y_three_class.shape == predictions.shape, "y_three_class must have the same shape as predictions."
        unique_preds = np.unique(predictions)

        if set(unique_preds.tolist()).issubset({0, 1}):
            cm = np.zeros((2, 3), dtype=int)
            for i in range(D):
                for j in range(N):
                    pred = predictions[i, j].astype(int)  
                    true = y_three_class[i, j].astype(int)
                    cm[pred, true] += 1
            print_confusion_matrix(
                cm,
                pred_labels=['0 (pred: not exists)', '1 (pred: exists)'],
                true_labels=['0 (GT: not mentioned)', '1 (GT: not exists)', '2 (GT: exists)']
            )

        elif set(unique_preds.tolist()).issubset({0, 1, 2}):
            
            cm = np.zeros((3, 3), dtype=int)
            for i in range(D):
                for j in range(N):
                    pred = predictions[i, j].astype(int)  # ensure it's an integer for indexing
                    true = y_three_class[i, j].astype(int)  # ensure it's an integer for indexing
                    # logger.info(f"pred: {pred}, true: {true}, mentioned: {mentioned[i,j]}")
                    cm[pred, true] += 1
            print_confusion_matrix(
                cm,
                pred_labels=[
                    '0 (pred: not mentioned)',
                    '1 (pred: not exists)',
                    '2 (pred: exists)'
                ],
                true_labels=[
                    '0 (GT: not mentioned)',
                    '1 (GT: not exists)',
                    '2 (GT: exists)'
                ]
            )
        else:
            raise ValueError("Predictions must contain only {0,1} or {0,1,2}")
        
        
        
        
class Mention_Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            
        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []
        # would be a list of T-long, each is a lits of MAX_NUM_OPERATIONS-long, for stratified accuracies        
        self.train_strat_acc_cont = []
        self.test_strat_acc_cont = []
        
        
        self.training_history = []
        self.test_history = []
        
    def flush_plot(self, save_path=None):
        # plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
        axs = axs.flat
        axs[0].plot(self.train_loss_cont, label="train")
        axs[0].plot(self.test_loss_cont, label="test")
        axs[0].set_title("Loss")
        axs[0].legend()
        axs[1].plot(self.train_acc_cont, label="train")
        axs[1].plot(self.test_acc_cont, label="test")
        axs[1].set_title("Accuracy")
        axs[1].legend()
        if save_path is not None:
            fig.savefig(save_path)
        # plt.show()
        # return a figure object
        

    def save_traces(self, ):
        tbd = {
            "train_loss_cont": self.train_loss_cont, "test_loss_cont" :self.test_loss_cont, 
            "train_acc_cont": self.train_acc_cont, "test_acc_cont": self.test_acc_cont, 
            "train_strat_acc_cont": self.train_strat_acc_cont, "test_strat_acc_cont": self.test_strat_acc_cont, 
        }
        with open(os.path.join(self.config.ckpt_path, "tensorboard.txt"), "w") as f:
            f.write(json.dumps(tbd) + "\n")
        with open(os.path.join(self.config.ckpt_path, "training_history.jsonl"), "w") as f:
            for record in self.training_history:
                f.write(json.dumps(record) + "\n")
        with open(os.path.join(self.config.ckpt_path, "test_history.jsonl"), "w") as f:
            for record in self.test_history:
                f.write(json.dumps(record) + "\n")

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(raw_model.state_dict(), os.path.join(self.config.ckpt_path, "checkpoint.ckpt"))

    def load_checkpoint(self):
        torch.load( os.path.join(self.config.ckpt_path, "checkpoint.ckpt"), map_location=self.device)

    def predict(self, prt=True):
        model, config = self.model, self.config
        model.train(False)
        data = self.test_dataset
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        
        pbar = enumerate(loader)
        losses = []
        totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
        hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        predictions = []
        gt_ls = []
        mentioned_ls = []
        
        results = [] # save predictions for further analysis, if needed
        
        for idx, (x, y, age, mentioned, removed) in pbar:
            gt_ls.append(y.cpu().numpy())
            mentioned_ls.append(mentioned.cpu().numpy())
            
            start_idx, end_idx = idx * config.batch_size, (idx + 1) * config.batch_size
            
            
            x = x.to(self.device)  # [B, f]
            y = y.to(self.device)  # [B, #task=64] 
            age = age.to(self.device)  # [B, #task=64], in 0--59
            mentioned = mentioned.to(self.device) # [B, #task], Bx100
            removed = removed.to(self.device) # [B, #task], Bx100, not used in the model, but can be used for analysis
            # y = mentioned # objective is to classify whether the object is mentioned or not, so just overwrite y with mentioned
            with torch.set_grad_enabled(False):
                logits, loss = model(x, y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)

                y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                hits = y_hat == y  # [B, #task]
                hits_nontrivial = (y_hat == y) * mentioned
                hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                predictions.append(y_hat.cpu().numpy())
                
        test_loss = float(np.mean(losses))
        test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
        test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()

        if prt: 
            logger.info(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%;  test non-triv acc {test_acc_nontriv*100:.2f}%")
        predictions_matrix = np.concatenate(predictions, axis=0)
        
        # need a confusion matrix for all classes, as well as trivial/non-trivial cases
        # we have prediction matrix (binary or tenary, not convertable), ground truth y(binary or ternary, convertable) and mentioned matrix
        # two cases, if it's a binary probe, we expand the confusion matrix to 2x3, if it's ternary, we just use 3x3
        self.predictions = predictions_matrix
        self.ground_truth = np.concatenate(gt_ls, axis=0)
        self.mentioned = np.concatenate(mentioned_ls, axis=0)
        # self.generate_confusion_matrix(
        #     predictions_matrix,
        #     np.concatenate(gt_ls, axis=0),
        #     np.concatenate(mentioned_ls, axis=0)
        # )
        return predictions_matrix
    
    
    
   
    def train(self, prt=True):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, scheduler = raw_model.configure_optimizers(config)
        
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
            hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_trivial_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_triv = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_removed_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_removed = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_mention_not_removed_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_mention_not_removed = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            
            
            recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FN)
            precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FP)
            total_recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FN 
            total_precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FP
            
            predictions = []
            gt_ls = []
            mentioned_ls = []
            pbar = tqdm(enumerate(loader), total=len(loader), disable=not prt) if is_train else enumerate(loader)
            num_mentioned_removed = 0
            num_mentioned_not_removed = 0
            total_mentioned = 0
            results = [] # save predictions for further analysis, if needed
            for it, (x, y, age, mentioned, removed) in pbar: # self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index]
                x = x.to(self.device)  # [B, f]
                y = y.to(self.device)  # [B, #task=64] 
                age = age.to(self.device)  # [B, #task=64], in 0--59
                mentioned = mentioned.to(self.device) # [B, #task]
                removed = removed.to(self.device)
                mentioned_not_removed = mentioned * (1 - removed)  # [B, #task], only counts the mentioned cases that are not removed
                y = mentioned  # objective is to classify whether the object is mentioned or not, so just overwrite y with mentioned
                start_idx, end_idx = it * config.batch_size, (it + 1) * config.batch_size
                # TODO why when this line is commented out, the acc is much highter? It should be the same as the binary case, but it is not.
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_triv += np.array([torch.sum((age == i) * (1 - mentioned)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_removed += np.array([torch.sum((age == i) * removed).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                    expanded_result = [
                        {
                            "index": i + start_idx,
                            "y_hat": y_hat[i].cpu().numpy(),
                            "mentioned": mentioned[i].cpu().numpy(),
                        } for i in range(len(y_hat))
                    ]
                    results.extend(expanded_result)  # save the predictions for further analysis
                    hits = y_hat == y  # [B, #task]
                    # ONLY COUNTS REMOVED CASES
                    hits_removed = (y_hat == y) * removed  # [B, #task], only counts the removed cases
                    hits_removed_epoch += np.array([torch.sum(hits_removed * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    hits_mention_not_removed = (y_hat == y) * mentioned_not_removed  # [B, #task], only counts the mentioned cases that are not removed
                    totals_epoch_mention_not_removed += np.array([torch.sum((age == i) * mentioned_not_removed).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    hits_mention_not_removed_epoch += np.array([torch.sum(hits_mention_not_removed * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    
                    
                    
                    
                    hits_nontrivial = (y_hat == y) * mentioned
                    hits_trivial = hits * (1 - mentioned)  # hits that are trivial (not mentioned)
                    hits_trivial_epoch += np.array([torch.sum(hits_trivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range
                                            (MAX_NUM_OPERATIONS)]).astype(float)
                    hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    predictions.append(y_hat.cpu().numpy())
                    gt_ls.append(y.cpu().numpy())
                    mentioned_ls.append(mentioned.cpu().numpy())
                    total_mentioned += np.sum(mentioned.cpu().numpy())
                    num_mentioned_removed += np.sum(removed.cpu().numpy())
                    num_mentioned_not_removed += np.sum(mentioned_not_removed.cpu().numpy())
                    recalls_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    precision_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_recalls_epoch += np.array([torch.sum((y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_precision_epoch += np.array([torch.sum((y_hat == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    # log basic stats of the distributions
               

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    mean_loss = float(np.mean(losses))
                    mean_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                    mean_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()
                    mean_acc_removed = np.sum(hits_removed_epoch).item() / np.sum(totals_epoch_removed).item()
                    mean_acc_mention_not_removed = np.sum(hits_mention_not_removed_epoch).item() / np.sum(totals_epoch_mention_not_removed).item()
                    mean_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                    mean_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                    

                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(f"epoch {epoch+1}: train loss {mean_loss:.5f}; lr {lr:.2e}; train acc {mean_acc*100:.2f}%; train non-triv acc: {mean_acc_nontriv*100:.2f}; trivial acc: {np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item()*100:.2f}%; removed acc: {mean_acc_removed*100:.2f}%; mention not removed acc: {mean_acc_mention_not_removed*100:.2f}%, recall {mean_recall*100:.2f}%; precision {mean_precision*100:.2f}%")
                    
                    self.training_history.append({
                        "epoch": epoch+1,
                        "train_loss": mean_loss,
                        "train_acc": mean_acc,
                        "train_nontriv_acc": mean_acc_nontriv,
                        "train_triv_acc": np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item() if np.sum(totals_epoch_triv).item() > 0 else 0.0,
                        "train_recall": mean_recall,
                        "train_precision": mean_precision
                    })
                    
                    
                    
            if is_train:
                self.train_loss_cont.append(mean_loss)
                self.train_acc_cont.append(mean_acc)
                self.train_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
                logger.info(f"Training Total mentioned: {total_mentioned}, num mentioned removed: {num_mentioned_removed}, num mentioned not removed: {num_mentioned_not_removed}")

            if not is_train:
                test_loss = float(np.mean(losses))
                scheduler.step(test_loss)
                test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()
                test_acc_trivial = np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item()
                test_acc_removed = np.sum(hits_removed_epoch).item() / np.sum(totals_epoch_removed).item()
                test_acc_mention_not_removed = np.sum(hits_mention_not_removed_epoch).item() / np.sum(totals_epoch_mention_not_removed).item()
                
                test_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                test_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                logger.info(f"Testing Total mentioned: {total_mentioned}, num mentioned removed: {num_mentioned_removed}, num mentioned not removed: {num_mentioned_not_removed}")
                if prt:                     
                    logger.info(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%;  test non-triv acc {test_acc_nontriv*100:.2f}%;    test trivial acc {test_acc_trivial*100:.2f}%; test removed acc: {test_acc_removed*100:.2f}%; test mention not removed acc: {test_acc_mention_not_removed*100:.2f}%; test recall: {test_recall*100:.2f}%; test precision: {test_precision*100:.2f}%")
                    self.test_history.append({
                    "epoch": epoch+1,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_nontriv_acc": test_acc_nontriv,
                    "test_triv_acc": test_acc_trivial,
                    "test_recall": test_recall,
                    "test_precision": test_precision
                })
                self.test_loss_cont.append(test_loss)
                self.test_acc_cont.append(test_acc)
                self.test_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
                predictions_matrix = np.concatenate(predictions, axis=0)
                self.predictions = predictions_matrix
                self.ground_truth = np.concatenate(gt_ls, axis=0)
                self.mentioned = np.concatenate(mentioned_ls, axis=0)
                
                return test_loss, predictions_matrix

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss, predictions_matrix = run_epoch('test')
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()
                
        # return predictions after last epoch
        return predictions_matrix
    
    def generate_confusion_matrix(self, file_path=None):
        """
        Generate a confusion matrix for the predictions against the ground truth.
        If mentioned is provided, it will be used to filter out trivial cases.
        predictions: numpy array of shape [N, N_Obj] (binary or ternary)
        ground_truth: numpy array of shape [N, N_Obj] (binary or ternary)
        mentioned: numpy array of shape [N, N_Obj] (binary), optional
        """
        
        def print_confusion_matrix(cm, pred_labels, true_labels, file_path=file_path):
            with open(file_path, "w") as f:
                f.write("Confusion Matrix:\n")
                f.write(" " * 22 + " | " + " | ".join(f"{label:^20}" for label in true_labels) + " |\n")
                f.write("-" * (25 + len(true_labels) * 24) + "\n")
                for i, row in enumerate(cm):
                    row_str = " | ".join(f"{val:^20}" for val in row)
                    f.write(f"{pred_labels[i]:<22} | {row_str} |\n")
                f.write("-" * (25 + len(true_labels) * 24) + "\n")
                
        predictions = self.predictions
        ground_truth = self.ground_truth
        mentioned = self.mentioned 
        D, N = predictions.shape
        # logging.info(f"predictions shape: {predictions.shape}, ground_truth shape: {ground_truth.shape}, mentioned shape: {mentioned.shape}")
        assert D == ground_truth.shape[0] and N == ground_truth.shape[1], "Predictions and ground truth must have the same shape."
        assert D == mentioned.shape[0] and N == mentioned.shape[1], "Predictions and mentioned must have the same shape."
        
        assert np.all(np.isin(predictions, [0, 1, 2])), "Predictions must be binary or ternary (0, 1, or 2)."
        assert np.all(np.isin(ground_truth, [0, 1, 2])), "Ground truth must be binary or ternary (0, 1, or 2)."
        
        y_three_class = np.full_like(ground_truth, fill_value=0)  
        num_class = predictions.max() + 1  # 0 or 1 or 2
        y_three_class[mentioned == 1] = ground_truth[mentioned == 1] + 1 if num_class == 2 else ground_truth[mentioned == 1]# y ∈ {0,1} → {1,2}
        assert y_three_class.shape == ground_truth.shape, "y_three_class must have the same shape as ground_truth."
        assert y_three_class.shape == predictions.shape, "y_three_class must have the same shape as predictions."
        unique_preds = np.unique(predictions)

        if set(unique_preds.tolist()).issubset({0, 1}):
            cm = np.zeros((2, 3), dtype=int)
            for i in range(D):
                for j in range(N):
                    pred = predictions[i, j].astype(int)  
                    true = y_three_class[i, j].astype(int)
                    cm[pred, true] += 1
            print_confusion_matrix(
                cm,
                pred_labels=['0 (pred: not exists)', '1 (pred: exists)'],
                true_labels=['0 (GT: not mentioned)', '1 (GT: not exists)', '2 (GT: exists)']
            )

        elif set(unique_preds.tolist()).issubset({0, 1, 2}):
            
            cm = np.zeros((3, 3), dtype=int)
            for i in range(D):
                for j in range(N):
                    pred = predictions[i, j].astype(int)  # ensure it's an integer for indexing
                    true = y_three_class[i, j].astype(int)  # ensure it's an integer for indexing
                    # logger.info(f"pred: {pred}, true: {true}, mentioned: {mentioned[i,j]}")
                    cm[pred, true] += 1
            print_confusion_matrix(
                cm,
                pred_labels=[
                    '0 (pred: not mentioned)',
                    '1 (pred: not exists)',
                    '2 (pred: exists)'
                ],
                true_labels=[
                    '0 (GT: not mentioned)',
                    '1 (GT: not exists)',
                    '2 (GT: exists)'
                ]
            )
        else:
            raise ValueError("Predictions must contain only {0,1} or {0,1,2}")
        
        
    