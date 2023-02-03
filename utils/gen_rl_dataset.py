import json
import torch
from torch.utils.data import Dataset

class GenRLDataset(Dataset):

    def __init__(self, preds, targets):

        assert len(preds) == len(targets)
        self.preds = preds
        self.targets = targets
    
    def __len__(self):
        return len(self.preds)
    
    def __getitem__(self, idx):
        pred = self.preds[idx]
        target = self.targets[idx]
        return {
            "pred_action": pred["action"],
            "pred_topic": pred["topic"],
            "target_action": target["action"],
            "target_topic": target["topic"]
        }
    
    def collate_fn(self, batch):
        ## return a list of dictionaries
        return batch