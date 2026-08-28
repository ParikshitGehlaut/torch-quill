import math
import torch
from torchquill.model.model_args import DeepSeekV3ModelArgs

def precompute_freqs_cis(args : DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    Precompute the frequencies for the RoPE (Rotary Positional Embedding) mechanism.

    Args:
        args (DeepSeekV3ModelArgs): The model arguments containing the necessary parameters.
    """

    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    # Basic RoPE frequency calculation
    # 10000 ** ((-2) * (i-1) / d) for i in range(1, d/2 + 1)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Create positional indices
    t = torch.arange(seqlen, dtype=torch.float32)

    # Outer Product : Positions x Frequencies
    freqs = torch.outer(t, freqs) # [seqlen, dim/2]

    # Convert to complex exponential form: e^(i * freqs * pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x : torch.Tensor, freqs_cis : torch.Tensor) -> torch.Tensor:
    """
    Apply the RoPE (Rotary Positional Embedding) to the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape [batch_size, seq_len, num_heads, head_dim].
        freqs_cis (torch.Tensor): The precomputed frequencies in complex form of shape [seq_len, head_dim/2].

    Returns:
        torch.Tensor: The tensor after applying RoPE, with the same shape as the input.
    """
    # Ensure the input tensor has the correct shape
    assert x.ndim == 4, "Input tensor must be 4-dimensional [batch_size, seq_len, num_heads, head_dim]"
    
    # x : [B, S, H, D] --> [B, S, H, D/2, 2] (split into real and imaginary parts)
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    # freqs_cis : [S, D/2] --> [1, S, 1, D/2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    y = torch.view_as_real(x * freqs_cis).flatten(3) # .flatten(3) : flattens dimensions starting from dimension 3

    return y.to(dtype)

