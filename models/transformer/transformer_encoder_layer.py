from torch import nn
from typing import Union, Callable

from torch import Tensor
import torch.nn.functional as F
from multi_head_attention import MultiheadAttention
from torch.nn import TransformerEncoderLayer as TorchTransformerEncoderLayer


class TransformerEncoderLayer(TorchTransformerEncoderLayer):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            norm_first: bool = False,
            device=None,
            dtype=None,
            attn_type: str = None
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, attn_type=attn_type)
