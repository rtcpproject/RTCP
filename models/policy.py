import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import MultiheadAttention
import torch.nn.functional as F

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.contiguous().view(-1, size[-1])).view(size)

class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x
    

class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        dropout = 0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size

        self.attention = MultiheadAttention(
            self.dim, 
            n_heads, 
            dropout=dropout, 
            batch_first=True, 
        )

    def forward(self, q, k, v, mask):
        output, _ = self.attention(q, k, v, key_padding_mask =mask)
        return output
    

class CrossEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        
        self.output_attention = CrossAttentionLayer(
            n_heads, 
            embedding_size, 
            ffn_size = ffn_size,
            dropout=attention_dropout, 
        )
        
        self.context_attention = CrossAttentionLayer(
            n_heads, 
            embedding_size, 
            ffn_size = ffn_size,
            dropout=attention_dropout, 
        )

        self.knowledge_attention = CrossAttentionLayer(
            n_heads, 
            embedding_size, 
            ffn_size = ffn_size,
            dropout=attention_dropout, 
        )

        self.path_attention = CrossAttentionLayer(
            n_heads,
            embedding_size, 
            ffn_size = ffn_size,
            dropout=attention_dropout, 
        )

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

        self.ffn_o = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn_c = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn_p = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn_k = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)


    def forward(self, o, c, p, k, c_mask, p_mask, k_mask):

        ### o output from the previous layer
        ### k knowledge latents
        ### c context latents
        ### p profile latents

        ### multi-head attention for context
        ### attent to it self.
        o = self.output_attention(o, o, o, (1 - p_mask).bool())

        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_o(o))
        o = _normalize(o, self.norm2)
        # o *= c_mask.unsqueeze(-1).type_as(o)
        
        ### cross multihead attention for path
        o = self.path_attention(o, p, p, (1 - p_mask).bool())
        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_p(o))
        o = _normalize(o, self.norm2)
        
        ### cross_multi_head_attention for context 
        o = self.context_attention(o, c, c, (1 - c_mask).bool())
        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_c(o))
        o = _normalize(o, self.norm2)
        # o *= p_mask.unsqueeze(-1).type_as(o)

        ### cross_multi_head_attention for output and knowledge
        o = self.knowledge_attention(o, k, k,  (1 - k_mask).bool())
        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_k(o))
        o = _normalize(o, self.norm2)
        # o *= k_mask.unsqueeze(-1).type_as(o)

        return o

class CrossEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
    
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(CrossEncoderLayer(
                n_heads,
                embedding_size,
                ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, inputs):

        ### inputs = [context_latents, knowledge_latents, profile_latents, context_mask, knowledge_mask, profile_mask]
        ### initialize the output with the path latent
        x = inputs['path_latent']

        ### forward pass
        for layer in self.layers:
            ### forward the input through each layer
            x = layer(
                x,
                inputs['context_latent'],
                inputs['path_latent'],
                inputs['knowledge_latent'],
                inputs['context_mask'],
                inputs['path_mask'],
                inputs['knowledge_mask']
            )

        return x


class PolicyModel(nn.Module):

    def __init__(self, context_encoder, knowledge_encoder, path_encoder, n_layers, n_heads, lm_hidden_size, ffn_size, fc_hidden_size, n_goals, n_topics, attention_dropout =0.2, relu_dropout = 0.2,  drop_out = 0.5):

        super().__init__()

        self.fc_hidden_size = fc_hidden_size
        self.context_encoder = context_encoder
        self.knowledge_encoder = knowledge_encoder
        self.path_encoder = path_encoder

        self.cross_encoder_model = CrossEncoder(
                    n_layers,
                    n_heads,
                    lm_hidden_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=drop_out,
        )

        
        self.goal_fc = nn.Linear(lm_hidden_size, fc_hidden_size)
        self.goal_out_layer = nn.Linear(fc_hidden_size, n_goals)

        self.goal_embedding = nn.Embedding(n_goals, fc_hidden_size)

        self.topic_fc = nn.Linear(lm_hidden_size, fc_hidden_size)
        ### with out goal information
        self.topic_out_layer = nn.Linear(fc_hidden_size, n_topics)

        # self.topic_out_layer = nn.Linear(2 * fc_hidden_size, n_topics)



    def forward(self, inputs):
        
        context_latents = self.context_encoder(input_ids = inputs['conversation'][0],
                                token_type_ids = inputs['conversation'][1],
                                position_ids = inputs['conversation'][2], 
                                attention_mask = inputs['conversation'][3], 
                                )[0]

        knowledge_latents = self.knowledge_encoder(input_ids = inputs['knowledge'][0],
                                token_type_ids = inputs['knowledge'][1],
                                position_ids = inputs['knowledge'][2], 
                                attention_mask = inputs['knowledge'][3], 
                                )[0]


        path_latents =  self.path_encoder(input_ids = inputs['path'][0],
                                token_type_ids = inputs['path'][1],
                                position_ids = inputs['path'][2], 
                                attention_mask = inputs['path'][3], 
                                )[0]


        out_dict = {
            "context_latent": context_latents,
            "knowledge_latent": knowledge_latents,
            "path_latent": path_latents,
            "context_mask": inputs['conversation'][3],
            "knowledge_mask": inputs['knowledge'][3],
            "path_mask": inputs['path'][3]
        }    

        output = self.cross_encoder_model(out_dict)
        cls_tokens = output[:,0,:]
        goal_logits = self.goal_out_layer(torch.relu(self.goal_fc(cls_tokens)))

        ### goal prediction loss and accuracy
        ce_loss = CrossEntropyLoss()
        goal_loss = ce_loss(goal_logits, inputs['next_goal'])
        goal_pred = torch.softmax(goal_logits, -1)

        _, pred_goal = goal_pred.max(-1)
        goal_acc = (torch.eq(pred_goal, inputs['next_goal']).float()).sum().item()

        # ### topic prediction loss and accuracy
        ### with goal information
        # goal_embeded = self.goal_embedding(pred_goal)
        # topic_logits = self.topic_out_layer(torch.cat([torch.relu(self.topic_fc(cls_tokens)), goal_embeded], dim =-1))

        ### without goal information
        topic_logits = self.topic_out_layer(torch.relu(self.topic_fc(cls_tokens)))

        topic_loss = ce_loss(topic_logits, inputs['next_topic'])
        topic_pred = torch.softmax(topic_logits, -1)

        _, pred_topic = topic_pred.max(-1)
        topic_acc = (torch.eq(pred_topic, inputs['next_topic']).float()).sum().item()

        output = {
            "goal_logits": goal_logits,
            "topics_logits": topic_logits,
            "loss": goal_loss + topic_loss,
            "acc": goal_acc,
            "total_tokens": pred_goal.shape[0],
            "topic_acc": topic_acc,
        }
        return output