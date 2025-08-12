import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        
        # Y 값 가져오기
        self.labels = dataset.Y  # torch.Tensor
        self.indices_0 = torch.where(self.labels == 0)[0].tolist()  # Y=0 인덱스
        self.indices_1 = torch.where(self.labels == 1)[0].tolist()  # Y=1 인덱스

        # 전체 데이터 개수와 배치 개수 계산
        self.total_samples = len(self.labels)
        self.num_batches = math.ceil(self.total_samples / self.batch_size)
        # print('self.num_batches: ', self.num_batches) # 73
        
        # 전체 데이터셋에서의 레이블 비율 계산
        num_0 = len(self.indices_0)
        num_1 = len(self.indices_1)
        # print('num_0, num_1: ', num_0, num_1) # 505 78
        total = num_0 + num_1
        # print('total: ', total) # 583
        
        self.p_0 = num_0 / total  # Y=0 비율
        self.p_1 = num_1 / total  # Y=1 비율
        
        # 각 배치에서 Y=0, Y=1을 유동적으로 샘플링
        self.batch_counts = [(math.ceil(self.batch_size * self.p_0), self.batch_size - math.ceil(self.batch_size * self.p_0))
                             for _ in range(self.num_batches)]
        # print('self.batch_counts: ', len(self.batch_counts), self.batch_counts) # 73

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices_0)
            random.shuffle(self.indices_1)

        indices_0 = self.indices_0.copy()
        indices_1 = self.indices_1.copy()

        batches = []
        for batch_0, batch_1 in self.batch_counts:
            batch = []
            
            if len(indices_0) >= batch_0:
                batch.extend(indices_0[:batch_0])
                indices_0 = indices_0[batch_0:]
            else:
                batch.extend(indices_0)
                indices_0 = []

            if len(indices_1) >= batch_1:
                batch.extend(indices_1[:batch_1])
                indices_1 = indices_1[batch_1:]
            else:
                batch.extend(indices_1)
                indices_1 = []
            random.shuffle(batch)
            batches.append(batch)

        # print('len(batches): ', len(batches)) # 73
        return iter(batches)

    def __len__(self):
        return self.num_batches

class Dataset(Dataset):
    def __init__(self, X, Y, Age, Norm_Age):
        self.X = X
        self.Y = Y# pMCI=1, sMCI=0 (size: 1)
        self.Age = Age
        self.Norm_Age = Norm_Age

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Age[idx], self.Norm_Age[idx]

def load_dataset(data_path, device):
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    # Load train data
    X_train = [torch.load(os.path.join(train_dir, 'X', f)) for f in sorted(os.listdir(os.path.join(train_dir, 'X')))]
    Y_train = [torch.load(os.path.join(train_dir, 'Y', f)) for f in sorted(os.listdir(os.path.join(train_dir, 'Y')))]
    Age_train = [torch.load(os.path.join(train_dir, 'Age', f)) for f in sorted(os.listdir(os.path.join(train_dir, 'Age')))]

    # Load test data
    X_test = [torch.load(os.path.join(test_dir, 'X', f)) for f in sorted(os.listdir(os.path.join(test_dir, 'X')))]
    Y_test = [torch.load(os.path.join(test_dir, 'Y', f)) for f in sorted(os.listdir(os.path.join(test_dir, 'Y')))]
    Age_test = [torch.load(os.path.join(test_dir, 'Age', f)) for f in sorted(os.listdir(os.path.join(test_dir, 'Age')))]

    # **Convert lists to tensors before passing to Dataset**
    X_train = torch.stack(X_train).to(device).float()  # (num_samples, 2, 148)
    Y_train = torch.stack(Y_train).to(device)  # (num_samples)
    Age_train = torch.stack(Age_train).to(device).float()  # (num_samples, 2)
    # print(X_train.shape, Y_train.shape, Age_train.shape) # torch.Size([583, 2, 148]) torch.Size([583]) torch.Size([583, 2])

    X_test = torch.stack(X_test).to(device).float()
    Y_test = torch.stack(Y_test).to(device)
    Age_test = torch.stack(Age_test).to(device).float()

    Age_all = torch.cat([Age_train, Age_test], dim=0)  # (train_size + test_size, 2)
    
    # **Min-Max Normalization**
    min_age, max_age = Age_all.min(), Age_all.max()
    Norm_Age_all = (Age_all - min_age) / (max_age - min_age)  # Normalize between 0 and 1

    Norm_Age_train = Norm_Age_all[:Age_train.size(0)]
    Norm_Age_test = Norm_Age_all[Age_train.size(0):]

    # Create dataset
    train_dataset = Dataset(X_train, Y_train, Age_train, Norm_Age_train)
    test_dataset = Dataset(X_test, Y_test, Age_test, Norm_Age_test)

    print('# train:', X_train.shape[0], '# test:', X_test.shape[0])

    return train_dataset, test_dataset, min_age, max_age

