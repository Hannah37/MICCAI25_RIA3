import torch
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

import os
import sys
import math
import wandb
from tqdm.auto import tqdm

import model

class Trainer(object):
    def __init__(
        self,
        device: torch.device('cuda:0'),
        network: model.Network,
        train_loader: DataLoader,
        test_loader: DataLoader,
        adv_w = 1.0,
        lr_lambda = 0.995,
        lr = 1e-4,
        batch_size = 16,
        epoch = 1000,
        optim = 'SGD',
        classifier_type = 'MLP',
        data = 'Amyloid',
        is_save = True
    ):
        super().__init__()

        self.device = device
        self.data = data
        self.is_save = is_save
        self.classifier = classifier_type
        self.network = network.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = epoch
        self.lr = lr
        self.lr_lambda = lr_lambda
        self.batch_size = batch_size
        self.adv_w = adv_w
        num_train_data = len(self.train_loader.dataset)
        self.niters_per_epoch = int(math.ceil(num_train_data / batch_size))
        

        if optim == 'SGD':
            self.opt = SGD(self.network.parameters(), lr=self.lr)
        elif optim == 'Adam':
            self.opt = Adam(self.network.parameters(), lr=self.lr)    

    def eval(self):
        self.network.eval()
        with torch.no_grad():  # Disable gradients for evaluation
            X, Y, _, Norm_Age = next(iter(self.test_loader))  # Unpack a single batch
            labels = Y.view(-1)  # Flatten labels

            logits = self.network.classifier(X[:, 0, :], Norm_Age[:, 0]) # only predict baseline data
            preds = torch.argmax(logits, dim=1)              

            accuracy = (preds == labels).sum().float() / labels.size(0)

            # Precision & Recall computation on GPU
            TP = ((preds == 1) & (labels == 1)).sum().float()  # True Positives
            FP = ((preds == 1) & (labels == 0)).sum().float()  # False Positives
            FN = ((preds == 0) & (labels == 1)).sum().float()  # False Negatives

            # Avoid division by zero
            precision = TP / (TP + FP + 1e-8)  # Precision = TP / (TP + FP)
            recall = TP / (TP + FN + 1e-8)  # Recall = TP / (TP + FN)

            F1 = 2 * (precision * recall) /(precision + recall + 1e-8)

        return accuracy.item(), precision.item(), recall.item(), F1.item()  

    def train(self):
        self.network.train()
        best_avg_results, best_epoch, best_acc, best_prec, best_recall, best_f1 = 0, 0, 0, 0, 0, 0
        self.scheduler = LambdaLR(optimizer=self.opt, lr_lambda=lambda epoch: self.lr_lambda ** epoch)
        best_model_dict = self.network.state_dict()
        
        for epoch in range(self.epoch):
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(self.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
            total_adv_loss, total_ce_loss, total_loss = 0.0, 0.0, 0.0
            
            # print(f"Train loader length: {len(self.train_loader)}") # 73
            for batch_idx, (X, Y, Age, norm_Age) in enumerate(self.train_loader):
                Y = Y.squeeze(-1)               
                pred, dist_given, dist_aug, gt = self.network(X, Y, Age, norm_Age)

                adv_loss = model.Loss.adversarial_loss(dist_given, dist_aug)
                ce_loss = model.Loss.cross_entropy_loss(pred, gt)
                L = adv_loss * self.adv_w + ce_loss

                self.opt.zero_grad()
                L.backward()
                self.opt.step()

                total_adv_loss += adv_loss.item()
                total_ce_loss += ce_loss.item()
                total_loss += L.item()

                pbar.set_description(f"Epoch {epoch + 1}/{self.epoch}: Adv = {adv_loss.item():.4f}, CE = {ce_loss.item():.4f}, Loss: {L.item():.4f}")
                pbar.update(1)
            pbar.close()
            self.scheduler.step()
            curr_lr = self.opt.param_groups[0]['lr']
            wandb.log({"lr": curr_lr})  

            avg_adv_loss = total_adv_loss / self.niters_per_epoch
            avg_ce_loss = total_ce_loss / self.niters_per_epoch
            avg_total_loss = total_loss / self.niters_per_epoch
            test_accuracy, test_precision, test_recall, test_F1 = self.eval()
            avg_results = (test_accuracy + test_precision + test_recall) / 3

            wandb.log({
                "Accuracy": test_accuracy,
                "Precision": test_precision,
                "Recall": test_recall,
                "Avg": avg_results,
                "F1" : test_F1,
                "Adv Loss": avg_adv_loss,
                "CE Loss": avg_ce_loss,
                "Total Loss": avg_total_loss
            })

            if avg_results > best_avg_results:
                best_avg_results, best_epoch, best_acc, best_prec, best_recall, best_f1 = avg_results, epoch, test_accuracy, test_precision, test_recall, test_F1
                results = {
                    "best_acc": test_accuracy,
                    "best_prec": test_precision,
                    "best_recall" : test_recall, 
                    "best_F1" : test_F1,
                    "best_avg" : best_avg_results,
                    "best_epoch" : best_epoch
                }
                wandb.config.update(results, allow_val_change=True)
                if self.is_save == True: 
                    fn = 'RA3_' + self.data + '_' + self.classifier + '_epoch_' + str(epoch + 1) + '_Acc_' + str(round(test_accuracy*100,2)) + '_F1_' + str(round(test_F1*100, 2)) + '.pt'
                    pth = os.path.join('saved_models', fn)
                    best_model_dict = self.network.state_dict()
                

            self.network.PGD.log_epsilons()

            print(f"Epoch {epoch + 1}/{self.epoch}: Avg = {avg_results*100:.2f}, Acc = {test_accuracy*100:.2f}, F1 = {test_F1*100:.2f}, Prec = {test_precision*100:.2f}, Recall = {test_recall*100:.2f}, "
              f"Loss = {avg_total_loss:.4f}, Adv = {avg_adv_loss:.4f}, CE = {avg_ce_loss:.4f}\n")

        if self.is_save == True: 
            torch.save(best_model_dict, pth)

        print(f"Best Epoch {best_epoch}: Best Avg = {best_avg_results*100:.2f}, Best Acc = {best_acc*100:.2f}, Best Prec = {best_prec*100:.2f}, Best Recall = {best_recall*100:.2f}, Best F1 = {best_f1*100:.2f}, "
                f"Loss = {avg_total_loss:.4f}\n")             