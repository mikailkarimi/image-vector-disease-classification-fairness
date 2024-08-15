from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import numpy as np


class VecDataset(Dataset):
    def __init__(self, info_df_path_mimic, data_path_mimic):
        self.PRED_LABEL= [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Lesion',
            'Lung Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.info_df_mimic = pd.read_csv(info_df_path_mimic)
        self.lables_mimic = self.info_df_mimic[self.PRED_LABEL]
        self.samples_path_mimic = self.info_df_mimic['path']
        self.data_path_mimic = data_path_mimic
        self.len_mimic = len(self.info_df_mimic)

    def __len__(self):
        return self.len_mimic
    
    def __getitem__(self, idx):
        path = self.samples_path_mimic.iloc[idx].split(".")[0]
        path = path + ".npy"
        full_item_path = self.data_path_mimic + path
        item = np.load(full_item_path)

        item = torch.from_numpy(item)

        label = self.lables_mimic.iloc[idx]
        label = list(label)
        label = torch.tensor(label, dtype=torch.float32)

        return {'data': item, 'labels': label}

