from functools import partial
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer


class PrefixTuningTemplate(nn.Module):
    r"""This is the implementation which support T5 and other Encoder-Decoder model,
    as soon as their blocks allows the ``past_key_values`` to be injected to the model.
    This implementation modifies the huggingface's T5 forward without touching the code-base.
    However, it may fail to work when used in DataParallel model. Please use it using
    single gpu or model-parallel training.
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained model.
        plm_config (:obj:`PretrainedConfig`): The configuration of the current pre-trained model.
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model.
        mapping_hook (:obj:`nn.Module`, optional):
        text (:obj:`str`, optional):
        num_token (:obj:`int`, optional):
        placeholder_mapping (:obj:`dict`):
        prefix_dropout (:obj:`float`, optional): The dropout rate for the prefix sequence.
    """
    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self,
                 config,
                 num_token: Optional[int] = 50,
                 prefix_dropout: Optional[float] = 0.0,
                 mid_dim: Optional[int] =  2048,
                 n_action_toks = 2,
                 n_topic_toks = 2,
                 use_goal_topic = 1
                ):
        super().__init__()
        self.num_token = num_token
        self.n_actions = 19
        self.n_topics = 646
        self.n_action_toks = n_action_toks
        self.n_topic_toks = n_topic_toks
        self.config = config
        self.use_goal_topic = use_goal_topic

        if isinstance(self.config, GPT2Config):
            self.n_decoder_layer = self.config.n_layer
            self.n_embd = self.config.n_embd
            self.n_head = self.config.n_head
            self.match_n_decoder_layer = self.n_decoder_layer
        else:
            raise Exception("Backbone model must be GPT2")

        self.mid_dim = mid_dim
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.generate_parameters() # in prefix tuning the template text has no interact with the parameters.

    
    def get_past_key_values(self, batch_size=1):
        pvs = []
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            temp_control = self.wte(input_tokens)
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
            _, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            pvs.append(past_key_values)
        else:
            pvs.append(None)

        ##### this branch if we use GPT2 as the PLM
        if (not self.config.is_encoder_decoder) or self.using_decoder_past_key_values:

            #### tasks_specific prompt tokens
            decoder_input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            #### project the task specific tokens to embedding vectors.
            decoder_temp_control = self.decoder_wte(decoder_input_tokens)
            ### applying a mlp layer.
            decoder_past_key_values = self.decoder_control_trans(decoder_temp_control) #bsz, seqlen, layer*emb
            _, decoder_seqlen, _ = decoder_past_key_values.shape

            #### IMPORTANT this code reshape the additional decoder tokens to the shape of the hidden vector computed by the PLM model.
            decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen, self.match_n_decoder_layer * 2, self.match_n_head,
                                                self.match_n_embd)
            
            #### appling drop out to the soft promots
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            #### reshape the soft prompt., split he prompt to 2 chunks.
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2) ## nlayer * 2, 
            pvs.append(decoder_past_key_values)

        else:
            pvs.append(None)
        return pvs

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False) # to allow automatic devicing
        ### embedding for projecting the soft tokens to embedding space.
        self.decoder_wte = nn.Embedding(self.num_token, self.n_embd)
        #### parametrization trick with a MIL layer.
        self.decoder_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)
        )

        ### parameters for the action-topic prompt
        self.input_action_tokens = nn.Parameter(torch.arange(self.n_action_toks * self.n_actions).long(), requires_grad=False) # to allow automatic devicing
        self.input_topic_tokens = nn.Parameter(torch.arange(self.n_topic_toks * self.n_topics).long(), requires_grad=False) # to allow automatic devicing
        self.action_decoder_wte = nn.Embedding(self.n_action_toks * self.n_actions, self.n_embd)
        self.topic_decoder_wte = nn.Embedding(self.n_topic_toks * self.n_topics, self.n_embd)
        self.action_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)
        )
        self.topic_control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            # nn.Linear(self.mid_dim, self.mid_dim),
            # nn.Tanh(),
            nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd)
        )

    def expand_to_batchsize(self, tup,  batch_size):
        return tuple(t.expand(-1, batch_size,-1,-1,-1) for t in tup)

    def expand_to_batchsize_for_layer(self, tup, batch_size, layer_id):
        return tup[layer_id].expand(-1, batch_size,-1,-1,-1)
    
    def compute_action_topic_prompt(self, batch):

        decoder_action_values = self.action_decoder_wte(self.input_action_tokens) # self.n_action_toks * self.n_actions * n_embedd
        decoder_topic_values = self.topic_decoder_wte(self.input_topic_tokens) # self.n_topic_toks * self.n_topics * n_embedd
        decoder_action_values = self.action_control_trans(decoder_action_values) 
        decoder_topic_values = self.topic_control_trans(decoder_topic_values) #(self.n_topic_toks * self.n_topics) * (self.n_decoder_layer * 2 * self.n_embd)
        b_action_values = []
        b_topic_values = []

        ### lookup action tokens
        for i in range(batch['action_id'].size(0)):
            a_id = batch['action_id'][i]
            t_id = batch['topic_id'][i]
            act_prompt = decoder_action_values[a_id: a_id + self.n_action_toks, :] ### 2 x (self.n_decoder_layer * 2 * self.n_embd)
            topic_prompt = decoder_topic_values[t_id: t_id + self.n_topic_toks, :] ### 2 x (self.n_decoder_layer * 2 * self.n_embd)
            b_action_values.append(act_prompt)
            b_topic_values.append(topic_prompt)
        
        b_action_values = torch.stack(b_action_values) # bs x 2 x (self.n_decoder_layer * 2 * self.n_embd)
        b_topic_values = torch.stack(b_topic_values)
        action_topic_prompt = torch.cat([b_action_values, b_topic_values ], dim = 1) # bs x 4 x (self.n_decoder_layer * 2 * self.n_embd)

        #### IMPORTANT this code reshape the additional decoder tokens to the shape of the hidden vector computed by the PLM model.
        action_topic_prompt = action_topic_prompt.view(-1, self.n_action_toks + self.n_topic_toks, self.match_n_decoder_layer * 2, self.match_n_head,
                                            self.match_n_embd)
        #### appling drop out to the soft promots
        action_topic_prompt = self.dropout(action_topic_prompt)
        #### reshape the soft prompt., split he prompt to 2 chunks.
        action_topic_prompt = action_topic_prompt.permute([2, 0, 3, 1, 4]).split(2) ## nlayer * 2, 
        return action_topic_prompt

    def forward(self, batch):
        r"""
        Convert input_ids to inputs_embeds
        for normal token, use the embedding inside PLM
        for new token, use MLP or LSTM
        """
        batch_size = batch['input_ids'].size(0)
        ### memorize some things here
        self.past_key_values = self.get_past_key_values()
        self.past_key_values = self.expand_to_batchsize(self.past_key_values[1], batch_size)
        if self.use_goal_topic:
            self.past_action_topic_values = self.compute_action_topic_prompt(batch)
            # past_key_values = (temp_1, temp_2)
            ## compute the action-topic prompt here
            past_key_values = []
            for a, b in zip(self.past_action_topic_values,self.past_key_values):
                temp = torch.cat([a, b], dim = -2)
                past_key_values.append(temp)
        else:
            past_key_values = self.past_key_values

        if 'attention_mask' in batch:
            am = batch['attention_mask']
            if self.use_goal_topic:
                batch['attention_mask'] = torch.cat([torch.ones((batch_size, self.num_token + self.n_action_toks + self.n_topic_toks), dtype = am.dtype,device=am.device), am], dim=-1)
            else:
                batch['attention_mask'] = torch.cat([torch.ones((batch_size, self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
        batch['past_key_values'] = tuple(past_key_values)
        return batch
