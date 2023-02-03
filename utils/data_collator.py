import torch


def max_seq_length(list_l):
    return max(len(l) for l in list_l)

def pad_sequence(list_l, max_len, padding_value=0):
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


class PlanCollator(object):
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
        up_ids = []
        hs_ids, hs_segs, hs_poss = [], [], []
        ks_ids, ks_segs, ks_poss = [], [], []
        ps_ids, ps_segs, ps_poss = [], [], []
        ts_ids, ts_segs, ts_poss = [], [], []

        n_topics, n_goals = [], []
        for sample in mini_batch:
            hs_ids.append(sample.conversation_ids)
            hs_segs.append(sample.conversation_segs)
            hs_poss.append(sample.conversation_poss)

            ks_ids.append(sample.knowledge_ids)
            ks_segs.append(sample.knowledge_segs)
            ks_poss.append(sample.knowledge_poss)

            ps_ids.append(sample.profile_ids)
            ps_segs.append(sample.profile_segs)
            ps_poss.append(sample.profile_poss)
            
            ts_ids.append(sample.path_ids)
            ts_segs.append(sample.path_segs)
            ts_poss.append(sample.path_poss)

            n_topics.append(sample.next_topic)
            n_goals.append(sample.next_goal)        
        
        batch_hs_ids = self.list_to_tensor(hs_ids)
        batch_hs_segs = self.list_to_tensor(hs_segs)
        batch_hs_poss = self.list_to_tensor(hs_poss)
        batch_hs_masks = self.get_attention_mask(batch_hs_ids)

        batch_ks_ids = self.list_to_tensor(ks_ids)
        batch_ks_segs = self.list_to_tensor(ks_segs)
        batch_ks_poss = self.list_to_tensor(ks_poss)
        batch_ks_masks = self.get_attention_mask(batch_ks_ids)

        batch_ps_ids = self.list_to_tensor(ps_ids)
        batch_ps_segs = self.list_to_tensor(ps_segs)
        batch_ps_poss = self.list_to_tensor(ps_poss)
        batch_ps_masks = self.get_attention_mask(batch_ps_ids)
        
        batch_ts_ids = self.list_to_tensor(ts_ids)
        batch_ts_segs = self.list_to_tensor(ts_segs)
        batch_ts_poss = self.list_to_tensor(ts_poss)
        batch_ts_masks = self.get_attention_mask(batch_ts_ids)

        collated_batch = {
            "conversation": [batch_hs_ids, batch_hs_segs, batch_hs_poss, batch_hs_masks],
            "knowledge":  [batch_ks_ids, batch_ks_segs, batch_ks_poss,  batch_ks_masks],
            "profile": [batch_ps_ids, batch_ps_segs, batch_ps_poss, batch_ps_masks  ],
            "path": [batch_ts_ids, batch_ts_segs, batch_ts_poss, batch_ts_masks],
            "next_goal": torch.LongTensor(n_goals).to(self.device),
            "next_topic": torch.LongTensor(n_topics).to(self.device)
        }

        return collated_batch