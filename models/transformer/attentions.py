import torch
from torch import nn
from torch.nn.functional import softmax


class DenseVanillaAttention(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super().__init__()
        self.w_1 = nn.Linear(in_dims, 64)
        self.w_2 = nn.Linear(64, sentence_length)
        self.relu = nn.ReLU()

        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        dense_attn = self.w_2(self.relu(self.w_1(query)))
        query = self.query_fc(query)
        key = self.key_fc(key).permute(0, 2, 1)

        attention = self.softmax((dense_attn + torch.bmm(query, key)))
        out = torch.bmm(attention, value)
        return out, attention


class RandomVanillaAttention(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super().__init__()
        self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query,key,value):
        query = self.query_fc(query)
        key = self.key_fc(key).permute(0, 2, 1)
        attention = self.softmax((self.attention + torch.bmm(query, key)))
        out = torch.bmm(attention, value)
        return out, attention


class FactorizedDenseAttention(nn.Module):
    def __init__(self, sentence_length, in_dims,  attn_dropout = 0.1):
        super(FactorizedDenseAttention, self).__init__()
        self.f = 1
        self.sentence_length = sentence_length
        self.f_a = nn.Linear(in_dims, self.f)
        self.f_b = nn.Linear(in_dims, sentence_length//self.f)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, sentence_length, len_q, mask=None, factorize=False):
        h_a = torch.repeat_interleave(self.f_a(q), self.sentence_length // self.f, -1)[:, :, :len_q]
        h_b = torch.repeat_interleave(self.f_b(q), self.f, -1)[:, :, :len_q]
        dense_attn = torch.matmul(h_a, h_b.transpose(2, 2))
        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)
        dense_attn = self.dropout(softmax(dense_attn, dim=-1))
        output = torch.bmm(dense_attn, v)
        return output, dense_attn


class FactorizedRandomAttention(nn.Module):

    def __init__( self,  batch_size, n_head, max_seq_len, attn_dropout = 0.1):
        super(FactorizedRandomAttention, self).__init__()

        self.random_attn_1 = torch.randn(batch_size, max_seq_len, 8, requires_grad = True)
        self.random_attn_2 = torch.randn(batch_size, 8, max_seq_len, requires_grad = True)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, v, len_q, mask=None, factorize=False):
        random_attn = torch.matmul(self.random_attn_1, self.random_attn_2)[:mask.shape[0],:len_q,:len_q]
        if mask is not None:
            random_attn = random_attn.masked_fill(mask == 0, -1e9)
        random_attn = self.dropout(softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)
        return output, random_attn


class RandomAttention(nn.Module):
    def __init__(self, batch_size, n_head, sentence_length, attn_dropout = 0.1):
        super(RandomAttention, self).__init__()
        self.random_attn = torch.randn(batch_size, sentence_length, sentence_length, requires_grad = False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, v, len_q, mask=None):
        random_attn = self.random_attn[:mask.shape[0],:len_q,:len_q]
        if mask is not None:
            random_attn = random_attn.masked_fill(mask == 0, -1e9)

        random_attn = self.dropout(softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)

        return output, random_attn


class DenseAttention(nn.Module):
    def __init__(self, in_dims, sentence_length ,attn_dropout = 0.1):
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(in_dims, 64)
        self.w_2 = nn.Linear(64, sentence_length)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None):
        dense_attn = self.w_2(self.relu(self.w_1(q)))[:, :, :len_q]
        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)
        dense_attn = self.dropout(softmax(dense_attn, dim=-1))
        output = torch.bmm(dense_attn, v)
        return output, dense_attn
