from typing import Optional, Union

import torch
import torch.nn as nn
import math

from .__utils import SwiGLU
from .Attention_module import MultiHeadAttention


class MoEGate(torch.nn.Module):
    def __init__(
        self,
        num_experts_per_tok: int,
        n_routed_experts: int,
        topk_method: str,
        n_group: int,
        topk_group: int,
        hidden_size: int,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.weight = torch.nn.Parameter(
            torch.empty((self.n_routed_experts, hidden_size))
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        batch, seq_len, h = x.shape
        hidden_states = x.view(-1, h)
        logits = torch.nn.functional.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(batch * seq_len, self.n_group, -1).max(dim=-1).values
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    batch * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(batch * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        return topk_idx, topk_weight


class MoE(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        topk_method: str,
        n_group: int,
        topk_group: int,
        hidden_dim: int | None = None,
        n_routed_experts: int = 12,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 2,
        mlp: str = "swiglu",
    ):
        super().__init__()
        self.experts_per_rank = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        mlp_block = SwiGLU
        self.hidden_dim = dim * 2 if hidden_dim is None else hidden_dim
        self.experts = torch.nn.ModuleList(
            [mlp_block(dim, hidden_dim) for i in range(n_routed_experts)]
        )
        self.gate = MoEGate(
            num_experts_per_tok, n_routed_experts, topk_method, n_group, topk_group, dim
        )
        self.shared_experts = mlp_block(dim, hidden_dim * n_shared_experts)

    def forward(self, x: torch.Tensor):
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        y = y.type(x.dtype)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=x.dtype)
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)

        y = y.view(*orig_shape)
        output = y + self.shared_experts(identity)
        return output


class Encoder_MoE_Attention_Block(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        #  ff_hidden_size:int,
        #  ff_dropout:float=0.3,
        num_experts_per_tok: int = 4,
        n_routed_experts: int = 15,
        topk_method: str = "group_limited_greedy",
        n_group: int = 5,
        topk_group: int = 4,
        hidden_dim: int = None,
    ):
        super(Encoder_MoE_Attention_Block, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        # self.ff_hidden_size = ff_hidden_size
        # self.ff_dropout = ff_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        if hidden_dim is None:
            hidden_dim = embed_size * 2
        else:
            self.hidden_dim = hidden_dim

        self.moe = MoE(
            embed_size,
            topk_method,
            n_group,
            topk_group,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
        )
        self.attention = MultiHeadAttention(embed_size, num_heads)
        # self.ff = nn.Sequential(
        #     nn.Linear(embed_size, ff_hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(ff_dropout),
        #     nn.Linear(ff_hidden_size, embed_size),
        #     nn.Dropout(ff_dropout)
        # )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.attention(x, mask)
        moe_output = self.moe(attention_output)
        # output = self.ff(attention_output + x)
        output = moe_output + x
        return output
