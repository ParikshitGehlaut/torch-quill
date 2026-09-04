import math
from typing import Optional

import torch
from torch import nn
from torchquill.model.attention import (
    ScaledDotProductAttentionWrapper,
)

from torchquill.model.model_args import DeepSeekV3ModelArgs
from torchquill.model.moe import FeedForward, MoE
from torchquill.model.rope import precompute_freqs_cis, apply_rotary_emb

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed forward layers.
    """

    def __init__(self, layer_id : int, model_args : DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    
    def forward(self, x : torch.Tensor, freqs_cis : torch.Tensor):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args : DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), presistent=False
        )

        self.layers = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def init_weights(
        self,
        init_std : Optional[float] = None,
        buffer_device : Optional[torch.device] = None,
        ):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        # Normal Distribution
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embedding.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device)

        if self.norm is not None:
            self.norm.reset_parameters()

        # We want var(y) ~ var(x), where y = W * x
        # Var(y) = n * (std of W)**2 * var(x)
        # std of W = 1 / (n)**0.5
        final_out_std = self.model_args.dim ** (-0.5) 
        cutoff_factor = 3

        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean = 0.0,
                std = final_out_std,
                a = -cutoff_factor * final_out_std,
                b = cutoff_factor * final_out_std,
            )

    def forward(self, tokens : torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h)
        output = self.output(h)
        return output

            
        

            
        
           
