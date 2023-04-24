from torch.utils.data import DataLoader,Dataset
import pandas as pd
from PIL import Image
import torch

class HAM10000(Dataset):
    """
    Custom PyTorch dataset for HAM10000 dataset
    - apply data augmentation for classes with low class support
    """
    def __init__(self, df: pd.DataFrame, augment_transform=None, basic_transform=None, data_col = 'path', label_col = 'cell_type_idx'):
        self.df = df
        self.augment_transform = augment_transform
        self.basic_transform = basic_transform
        self.data_col = data_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df[self.data_col][index])
        y = torch.tensor(int(self.df[self.label_col][index]))

        # if data needs to be augmented, apply augmentation transform
        if self.augment_transform:
            X = self.augment_transform(X)
        
        # if data does not need to be augmented (major class or test data), apply basic transform
        elif self.basic_transform:
            X = self.basic_transform(X)

        return X, y
     