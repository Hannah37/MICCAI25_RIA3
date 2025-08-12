import wandb
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

import backbones

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(149, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, x, age):
        h = torch.cat([x, age.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)
        return h

class Loss:
    @staticmethod
    def adversarial_loss(dist_given, dist_aug):
        d_total = dist_given - dist_aug
        return torch.mean(torch.norm(d_total, p=2, dim=-1)) 

    @staticmethod
    def cross_entropy_loss(pred, target):
        return nn.CrossEntropyLoss()(pred, target.long())

class Network(nn.Module):
    def __init__(self, hidden_dim, classifier_type, device, min_age, max_age, eps_init, is_clamp):
        super(Network, self).__init__()
        self.device = device
        self.min_age = min_age
        self.max_age = max_age
        self.is_clamp = is_clamp

        if classifier_type == 'MLP':
            self.classifier = backbones.MLP(hidden_dim=hidden_dim)
        elif classifier_type == 'TabNet':
            self.classifier = backbones.TabNet(n_d=hidden_dim, n_a=hidden_dim)
        elif classifier_type == 'FT':
            self.classifier = backbones.FT_Transformer(embed_dim=hidden_dim)
        elif classifier_type == 'NODE':
            self.classifier = backbones.NODE(tree_dim=hidden_dim)

        self.eps_min = eps_init
        self.epsilon_b = nn.Parameter(torch.full((148,), eps_init), requires_grad=True) 
        self.epsilon_f = nn.Parameter(torch.full((148,), eps_init), requires_grad=True) 
        self.PGD = PGD(self.classifier, self.device, self.epsilon_b, self.epsilon_f, is_clamp)
        
    def forward(self, X, Y, Age, Norm_Age):
        x_b, x_f = X[:, 0, :], X[:, 1, :] # (batch, 2, 148)
        age_b, age_f = Age[:, 0], Age[:, 1] # (batch, 2)
        norm_age_b, norm_age_f = Norm_Age[:, 0], Norm_Age[:, 1] # (batch, 2)

        y_b = torch.zeros_like(Y).to(self.device) # (batch_size, 1) # always MCI
        y_f = Y.to(self.device)  # (batch_size, 1) # pMCI&AD=1, sMCI&MCI=0
        
        '''ROI-Adaptive Adversarial Attack (RA^3)'''
        mask_pMCI = (Y == 1).view(-1)
        mask_sMCI = (Y == 0).view(-1)

        num_pmci = mask_pMCI.sum().item()  
        num_smci = mask_sMCI.sum().item()  
        num_aug = num_smci - num_pmci

        if (num_pmci > 0) and (num_aug > 0):  # if pMCI exists & (num_smci > num_pmci)
            x_b_adv_M, x_f_adv_M, x_b_aug, x_f_aug, age_b_aug, age_f_aug = self.PGD.attack(
                x_b[mask_pMCI], x_f[mask_pMCI], y_b[mask_pMCI], y_f[mask_pMCI], age_b[mask_pMCI], age_f[mask_pMCI], num_aug
            )
            d1 = x_b_adv_M - x_b[mask_pMCI]
            d2 = x_f[mask_pMCI] - x_f_adv_M 
            dist_aug = d1 + d2
            dist_given = x_f[mask_pMCI] - x_b[mask_pMCI]
                    
            # Combine original and augmented data
            x_b = torch.cat([x_b, x_b_aug], dim=0)
            norm_age_b_aug = (age_b_aug - self.min_age) / (self.max_age - self.min_age)
            norm_age_b = torch.cat([norm_age_b, norm_age_b_aug], dim=0)
            Y_aug = torch.ones(num_aug).to(self.device)
            Y = torch.cat([Y, Y_aug], dim=0)

            # Shuffle the combined dataset
            indices = torch.randperm(x_b.size(0)).to(x_b.device)
            x_b = x_b[indices]
            norm_age_b = norm_age_b[indices]
            Y = Y[indices]    

            '''pMCI/sMCI Classification'''
            pred = self.classifier(x_b, norm_age_b)
            return pred, dist_given, dist_aug, Y

        else: 
            # print(" No need to augment pMCI. #pMCI: ", num_pmci, " #sMCI: ", num_smci)
            pred = self.classifier(x_b, norm_age_b) # pMCI/sMCI Classification
            return pred, torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), Y

class PGD(nn.Module):
    def __init__(self, classifier, device, epsilon_b, epsilon_f, is_clamp):
        """
        PGD Adversarial Attack with Trainable Epsilon.
        Args:
            classifier: Target model for attack.
            epsilon: Step size for each attack iteration.
        """
        super(PGD, self).__init__()
        self.classifier = classifier
        self.device = device

        self.epsilon_b = epsilon_b
        self.epsilon_f = epsilon_f
        self.eps_min = self.epsilon_b[0].item()
        self.is_clamp = is_clamp

    def log_epsilons(self):
        """Log the first 3 elements of epsilon_b and epsilon_f to wandb."""
        if self.is_clamp:
            eps_b = self.epsilon_b.clamp(min=self.eps_min)
            eps_f = self.epsilon_f.clamp(min=self.eps_min)
        else:
            eps_b = torch.abs(self.epsilon_b)
            eps_f = torch.abs(self.epsilon_f)

        wandb.log({
            "eps_b_0": eps_b[0].item(),
            "eps_b_1": eps_b[1].item(),
            "eps_b_2": eps_b[2].item(),
            "eps_f_0": eps_f[0].item(),
            "eps_f_1": eps_f[1].item(),
            "eps_f_2": eps_f[2].item()
        })

    def get_attack_num(self, age_b, age_f):
        monthly_diffs = (age_f - age_b) * 12
        batch_steps = torch.round(monthly_diffs / 2).int()  # Round to nearest integer

        # Return the maximum num_steps in the batch
        max_steps = batch_steps.max().item()
        return max_steps, batch_steps

    def attack(self, x_b, x_f, y_b, y_f, age_b, age_f, num_aug):
        """
        Perform PGD attack on Baseline (towards AD=1) and Follow-up (towards MCI=0).
        """

        batch_size = x_b.shape[0]
        max_steps, batch_steps = self.get_attack_num(age_b, age_f)

        adv_b_list, adv_f_list = [], []
        age_b_list, age_f_list = [], []
        adv_b_final_list, adv_f_final_list = [], [] # last attacked data


        for i in range(batch_size):
            # Clone input tensors to create adversarial samples
            x_b_adv = x_b[i].clone().to(self.device).detach().requires_grad_(True)
            x_f_adv = x_f[i].clone().to(self.device).detach().requires_grad_(True)

            age_b_subj = age_b[i].unsqueeze(0)
            age_f_subj = age_f[i].unsqueeze(0)

            for step in range(batch_steps[i].item()): 
                outputs_b = self.classifier(x_b_adv.unsqueeze(0), age_b_subj)  
                outputs_f = self.classifier(x_f_adv.unsqueeze(0), age_f_subj)

                # Loss (Baseline → AD, Follow-up → MCI)
                loss_b = nn.CrossEntropyLoss()(outputs_b, torch.zeros_like(y_b[i].unsqueeze(0)))  # baseline: deviate from sMCI
                loss_f = -nn.CrossEntropyLoss()(outputs_f, torch.zeros_like(y_f[i].unsqueeze(0)))  # follow-up: towards sMCI

                # calculate Gradient 
                loss = loss_b + loss_f
                loss.backward(retain_graph=True)

                if self.is_clamp == True:
                    x_b_adv = x_b_adv + self.epsilon_b.clamp(min=self.eps_min) * x_b_adv.grad.sign()
                    x_f_adv = x_f_adv + self.epsilon_f.clamp(min=self.eps_min) * x_f_adv.grad.sign()
                else:
                    x_b_adv = x_b_adv + torch.abs(self.epsilon_b) * x_b_adv.grad.sign()
                    x_f_adv = x_f_adv + torch.abs(self.epsilon_f) * x_f_adv.grad.sign()

                x_b_adv.retain_grad()  
                x_f_adv.retain_grad()  

                # save attacked data
                adv_b_list.append(x_b_adv)
                adv_f_list.append(x_f_adv)

                # save ages of attacked data
                age_b_list.append(age_b[i] + (step + 1) / 12)  # Attack Baseline → Age increased
                age_f_list.append(age_f[i] - (step + 1) / 12)  # Attack Follow-up → Age decreased

                age_b_subj = (age_b[i] + (step + 1) / 12).unsqueeze(0)
                age_f_subj = (age_f[i] - (step + 1) / 12).unsqueeze(0)
    
            adv_b_final_list.append(x_b_adv)
            adv_f_final_list.append(x_f_adv)

        # Ramdomly sample augmented pMCI data
        sampled_steps = random.choices(range(len(adv_b_list)), k=num_aug)
        x_b_augmented = torch.stack([adv_b_list[m] for m in sampled_steps])
        x_f_augmented = torch.stack([adv_f_list[m] for m in sampled_steps])
        age_b_augmented = torch.stack([age_b_list[m] for m in sampled_steps])
        age_f_augmented = torch.stack([age_f_list[m] for m in sampled_steps])

        # Dimensionality correction before concatenation
        x_b_augmented = x_b_augmented.squeeze(1) if x_b_augmented.dim() == 3 else x_b_augmented
        x_f_augmented = x_f_augmented.squeeze(1) if x_f_augmented.dim() == 3 else x_f_augmented
        age_b_augmented = age_b_augmented.squeeze(1) if age_b_augmented.dim() == 2 else age_b_augmented
        age_f_augmented = age_f_augmented.squeeze(1) if age_f_augmented.dim() == 2 else age_f_augmented

        adv_b_final = torch.stack([t for t in adv_b_final_list]).to(self.device)
        adv_f_final = torch.stack([t for t in adv_f_final_list]).to(self.device)

        return adv_b_final, adv_f_final, x_b_augmented, x_f_augmented, age_b_augmented, age_f_augmented