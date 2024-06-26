import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# 上三角形为1，其他为0的mask
def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

def adjust_attention_scores(query, key, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return prob_attn

def attention(query, key, value, rel, l1, l2, timestamp, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    rel = rel * mask.to(torch.float) # future masking of correlation matrix.
    rel_attn = rel.masked_fill(rel == 0, -10000)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    # print('query shape', query.shape)
    # print('scores shape', scores.shape)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

        time_stamp= torch.exp(-torch.abs(timestamp.float()))
        #
        time_stamp=time_stamp.masked_fill(mask,-np.inf)


    prob_attn = F.softmax(scores, dim=-1)
    # print('prob_attn', prob_attn.shape)
    # print('value shape', value.shape)
    time_attn = F.softmax(time_stamp,dim=-1)
    prob_attn = (1-l2)*prob_attn+l2*time_attn
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)

    prob_attn = (1-l1)*prob_attn + (l1)*rel_attn
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, rel, l1, l2, pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings

    scores = torch.matmul(query, key.transpose(-2, -1))

    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, rel, l1, l2, timestamp, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)
        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, rel, l1, l2, timestamp, pos_key_embeds, pos_value_embeds,  mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn


class RKT(nn.Module):
    def __init__(self, num_items,  embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob):
        """Self-attentive knowledge tracing.
        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(RKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos

        self.item_embeds = nn.Embedding(num_items + 1, embed_size , padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, 1, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))

    def get_inputs(self, q, r):
        q = self.item_embeds(q)
        # skill_inputs = self.skill_embeds(skill_inputs)
        r = r.unsqueeze(-1).float()

        inputs = torch.cat([q, q], dim=-1)
        inputs[..., :self.embed_size] *= r
        inputs[..., self.embed_size:] *= 1 - r
        # [emb 0] 答对 [0 emb] 答错
        return inputs

    def get_query(self, nxt_seq):
        nxt_seq = self.item_embeds(nxt_seq)
        # skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([nxt_seq], dim=-1)
        return query

    def forward(self, q, r, nxt_seq, rel, timestamp):
        inputs = self.get_inputs(q, r) # (bs, sq_len, 2*emb_size)

        inputs = F.relu(self.lin_in(inputs)) # (bs, sq_len, emb_size)
        # get nxt_seq emb
        query = self.get_query(nxt_seq)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()
        outputs, attn  = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, timestamp, self.encode_pos,
                                                   self.pos_key_embeds, self.pos_value_embeds, mask)
        outputs = self.dropout(outputs)
        
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, self.encode_pos, timestamp, self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return self.lin_out(outputs), attn

    