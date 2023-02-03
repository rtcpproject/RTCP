import torch


def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


class RLPretrainingCollator(object):
    """
    Data collator for planning
    """
    def __init__(self, device, padding_idx=0):
        self.device = device
        self.padding_idx = padding_idx
    
    def list_to_tensor(self, list_l):
        max_len = max_seq_length(list_l)
        padded_lists = []
        for list_seq in list_l:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=self.padding_idx))
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor

    def varlist_to_tensor(self, list_vl):
        lens = []
        for list_l in list_vl:
            lens.append(max_seq_length(list_l))
        max_len = max(lens)
        
        padded_lists = []
        for list_seqs in list_vl:
            v_list = []
            for list_l in list_seqs:
                v_list.append(pad_sequence(list_l, max_len, padding_value=self.padding_idx))
            padded_lists.append(v_list)
        input_tensor = torch.tensor(padded_lists, dtype=torch.long)
        input_tensor = input_tensor.to(self.device).contiguous()
        return input_tensor
    
    def get_attention_mask(self, data_tensor: torch.tensor):
        attention_mask = data_tensor.masked_fill(data_tensor == self.padding_idx, 0)
        attention_mask = attention_mask.masked_fill(attention_mask != self.padding_idx, 1)
        attention_mask = attention_mask.to(self.device).contiguous()
        return attention_mask
    
    def custom_collate(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        ts_ids = []
        labels = []
        for sample in mini_batch:
            ts_ids.append(sample.path_ids)
            labels.append(sample.label)      
        
        batch_ts_ids = self.list_to_tensor(ts_ids)

        collated_batch = {
            "path": batch_ts_ids,
            "label": torch.LongTensor(labels).to(self.device),
        }

        return collated_batch