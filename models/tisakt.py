import torch
from torch import nn
import numpy as np
import sys
import os

import numpy as np
import torch

from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from .hgnn import HGNNGraph

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev
        self.l1 = nn.Parameter(torch.rand(1))

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V, time_attn):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case

        # attn_weights = self.l1 * attn_weights + (1- self.l1) * time_attn
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs


class TiSASRec(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, hidden_units, maxlen, time_span, num_heads, num_blocks, device, dropout_rate):
        super(TiSASRec, self).__init__()

        self.dev = device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # self.item_emb = torch.nn.Embedding(self.item_num+1, hidden_units, padding_idx=0)
        # self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(maxlen, hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(maxlen, hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(time_span+1, hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(time_span+1, hidden_units)

        # self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(hidden_units,
                                                            num_heads,
                                                            dropout_rate,
                                                            device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, seqs, log_seqs, time_matrices, mask, time_attn):
        '''
            seqs: seq的emb
            log_seqs: seq的id
            time_matrices ？
        '''
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # seqs *= self.item_emb.embedding_dim ** 0.5
        # seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        # print(self.dev)
        # time_matrices = torch.LongTensor(time_matrices.cpu())
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        # timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask = mask
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V, time_attn)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, mask, seq_emb, log_seqs, time_matrices, time_attn): # for training
        log_feats = self.seq2feats(seq_emb, log_seqs, time_matrices, mask, time_attn)
        return log_feats


class TiSAKT(nn.Module):
    def __init__(self, graph, num_items, device, embsize, maxlen, time_span, num_heads, num_blocks, dropout_rate):
        super(TiSAKT, self).__init__()
        self.device = device

        self.latent_dim = embsize  # 压缩后的题目emb维度
        self.response_dim = self.latent_dim * 2  # 包含答案的习题emb维度
        self.input_dim = self.latent_dim + self.response_dim
        

        self.hgnn_model = HGNNGraph(graph, self.latent_dim)
        self.question_emb = torch.nn.Embedding(num_items+1, self.latent_dim)
        self.response_emb = torch.nn.Embedding(2 * num_items+2, self.response_dim)
        self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.linear_tranform = nn.Linear(self.input_dim, self.latent_dim)
        self.fc = nn.Linear(self.latent_dim, self.latent_dim)  # 最后的输出emb
        self.attetion = TiSASRec(self.latent_dim, maxlen, time_span, num_heads, num_blocks, device, dropout_rate)

    def forward(self, seq, r, qry, time_matrices, mask):
        """
        :param seq: (bs, seq_len)
        :param ans: (bs, seq_len)
        :param seq_hidden: (bs, seq_len, latent_dim)
        :param nxtq_emb: (bs, 1)
        :return: pred: (bs, 1) y_pred (bs, seq_len, latent_dim)
        """
        # question_emb = self.hgnn_model()
        seq_hidden = self.question_emb(seq)
        # seq_hidden = question_emb[seq]
        seq_hidden = self.item_emb_dropout(seq_hidden)

        ans = seq * 2 + r
        seq_emb = torch.cat((seq_hidden, self.response_emb(ans)), dim=-1)
        seq_emb = self.linear_tranform(seq_emb)

        time_attn = torch.exp(-torch.abs(time_matrices.float()))
        time_attn = torch.exp(-torch.abs(time_matrices.float()))

        attn_out = self.attetion(mask, seq_emb, seq, time_matrices, time_attn)
        y_pred = self.fc(attn_out)
        
        pos_embs =  self.question_emb(qry)
        # pos_embs = question_emb[qry]
        pred = torch.sigmoid(torch.sum(torch.mul(y_pred, pos_embs), dim=-1))
       
        return pred
    
    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            loss_mean = []

            for data in train_loader:
                q, r, qshft, rshft, m, time, ans = data

                self.train()

                p = self(q.long(), r.long(), qshft.long(), time, m)
                p = torch.masked_select(p, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m, time, ans = data

                    self.eval()

                    p = self(q.long(), r.long(), qshft.long(), time, m)
                    p = torch.masked_select(p, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.cpu().numpy(), y_score=p.cpu().numpy()
                    )

                    acc =  accuracy_score(t.cpu().numpy(), np.array(p.cpu().numpy()) >= 0.5)
                    
                    loss_mean = np.mean(loss_mean)

                    print(
                        "Epoch: {},   AUC: {},   ACC:{},   Loss Mean: {}"
                        .format(i, auc, acc, loss_mean)
                    )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means