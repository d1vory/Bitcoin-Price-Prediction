from typing import Callable, List, Optional, Tuple
import math
import warnings
from torch import Tensor
import torch
from torch import _VF
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes
from typing import TYPE_CHECKING
from torch import nn
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.functional import _in_projection_packed, linear, dropout, softmax, _in_projection, pad


class DVMixtureSynthesizers(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super(DVMixtureSynthesizers, self).__init__()
        # Dense + Vanilla

        # Dense
        self.w_1 = nn.Linear(in_dims, 64)
        self.w_2 = nn.Linear(64, sentence_length)
        self.relu = nn.ReLU()

        # Vanilla
        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query,key,value):
        dense_attn=self.w_1(query)
        dense_attn=self.relu(dense_attn)
        dense_attn=self.w_2(dense_attn)
        query = self.query_fc(query)
        key = self.key_fc(key).permute(0, 2, 1)

        vanilla_energy = torch.bmm(query, key)
        energy = dense_attn + vanilla_energy
        attention = self.softmax(energy)
        out = torch.bmm(attention, value)

        return out, attention

class RVMixtureSynthesizers(nn.Module):
    def __init__(self, in_dims, sentence_length):
        super(RVMixtureSynthesizers, self).__init__()
        # Random + Vanilla

        # Random
        self.attention = nn.Parameter(torch.empty(1, sentence_length, sentence_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

        # Vanilla
        self.query_fc = nn.Linear(in_dims, in_dims)
        self.key_fc = nn.Linear(in_dims, in_dims)

        self.value_fc = nn.Linear(in_dims, in_dims)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query,key,value):
        query = self.query_fc(query)
        key = self.key_fc(key).permute(0, 2, 1)

        vanilla_energy = torch.bmm(query, key)
        energy = self.attention + vanilla_energy
        attention = self.softmax(energy)
        out = torch.bmm(attention, value)

        return out, attention




class FactorizedDenseAttention(nn.Module):
    def __init__(self, sentence_length, in_dims,  attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2
        super(FactorizedDenseAttention, self).__init__()
        self.f =1
        self.sentence_length = sentence_length
        self.f_a = nn.Linear(in_dims, self.f)
        self.f_b = nn.Linear(in_dims, sentence_length//self.f)

        #self.a = 1
        #self.b = sentence_length // self.a
        #self.a_proj = nn.Linear(in_dims, self.a).cpu()
        #self.b_proj = nn.Linear(in_dims, self.b).cpu()

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q,  v, sentence_length, len_q,mask=None, factorize=False):

        h_a = torch.repeat_interleave(self.f_a(q), self.sentence_length//self.f, -1)[:,:,:len_q]
        h_b = torch.repeat_interleave(self.f_b(q), self.f, -1)[:,:,:len_q]

        #h_a = self.f_a(q).repeat([1, 1, sentence_length//self.f])[:,:,:len_q].cpu()
        #h_b = self.f_b(q).repeat([1, 1, self.f])[:,:,:len_q].cpu()
        #dense_attn = torch.matmul(h_a, h_b.transpose(2, 2))
        #dense_attn = h_a * h_b
        dense_attn = torch.matmul(h_a , h_b.transpose(2, 2))

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
        # b x n x max_len x max_len -> b x n x lq x lq #[:,:,:len_q,:len_q]
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

        # b x n x max_len x max_len -> b x n x lq x lq
        random_attn = self.random_attn[:mask.shape[0],:len_q,:len_q]


        if mask is not None:
            random_attn = random_attn.masked_fill(mask == 0, -1e9)

        random_attn = self.dropout(softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)

        return output, random_attn

class DenseAttention(nn.Module):
    def __init__(self, in_dims, sentence_length ,attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(in_dims, 64)
        self.w_2 = nn.Linear(64, sentence_length)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q,mask=None):

        # b x n x lq x dq -> b x n x lq x lq #
        dense_attn = self.w_2(self.relu(self.w_1(q)))[:,:,:len_q]
        #print(dense_attn.shape)
        #dense_attn=self.w_1(q).cpu()
        #dense_attn=self.relu(dense_attn).cpu()
        #dense_attn=self.w_2(dense_attn).cpu()
       
        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)

        dense_attn = self.dropout(softmax(dense_attn, dim=-1))
        output = torch.bmm(dense_attn, v)
        
        return output, dense_attn



    
def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(

    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    attn_type: str=None
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    
    if attn_type=="dense":
        dense_attn=DenseAttention(in_dims=head_dim,sentence_length=src_len,attn_dropout=dropout_p)
        attn_output, attn_output_weights = dense_attn(q, v, len_q=tgt_len,mask=attn_mask)
        
    elif attn_type=="random":
        random_attn=RandomAttention(batch_size=bsz, n_head=num_heads, sentence_length=src_len, attn_dropout = dropout_p)
        attn_output, attn_output_weights=random_attn(v, len_q=tgt_len, mask=attn_mask)

    elif attn_type=="fac_random":
        fac_random=FactorizedRandomAttention(batch_size=head_dim,n_head=num_heads,max_seq_len=src_len,attn_dropout=dropout_p)
        attn_output, attn_output_weights = fac_random(v, len_q=tgt_len,mask=attn_mask) 
        
    elif attn_type=="fac_dense":
        fac_dense=FactorizedDenseAttention(in_dims=head_dim,sentence_length=src_len,attn_dropout=dropout_p)
        attn_output, attn_output_weights = fac_dense(q,v, sentence_length=src_len,len_q=tgt_len,mask=attn_mask) 
        
    elif attn_type=="rv_mix":
        RV_mix = RVMixtureSynthesizers(in_dims=head_dim,sentence_length=src_len)
        attn_output, attn_output_weights = RV_mix(q,k,v) 

    elif attn_type=="dv_mix":
        DV_mix = DVMixtureSynthesizers(in_dims=head_dim,sentence_length=src_len)
        attn_output, attn_output_weights = DV_mix(q,k,v) 
             
    else:
        attn_output, attn_output_weights = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)



    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
