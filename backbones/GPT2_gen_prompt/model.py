import torch
import torch.nn as nn


class PromptGPT2(nn.Module):

    def __init__(self, plm, prefix_model, freeze_plm = True):
        super(PromptGPT2, self).__init__()
        self.plm = plm
        self.prefix_model = prefix_model
        if freeze_plm:
            print("freeze all pretrained language model parameters.")
            for param in self.plm.parameters():
                param.requires_grad = False
    
    
    def forward(self, batch):
        batch = self.prefix_model(batch)
        batch = {
            "input_ids": batch["input_ids"],
            "labels": batch["labels"],
            "past_key_values": batch["past_key_values"]
        }
        output = self.plm(**batch, return_dict = True)
        return output