from dataclasses import dataclass, field
from torch import nn
from torchquill.model.moe_args import MoEArgs

@dataclass
class DeepSeekV3ModelArgs:
    max_seq_len : int = 4096 * 4
    vocab_size : int = 102400
    dim : int = 2048
    inter_dim : int = 10944
    moe_inter_dim : int = 1408
    n_layers : int = 27
    n_dense_layers : int = 1
    n_heads : int = 16
    norm_eps : float = 1e-5 # eps used for RMSNorm

    # MoE
    moe_args : MoEArgs = field(default_factory=MoEArgs)

    # MLA (Multi-head Latent Attention)
    q_lora_rank : int = 0
    kv_lora_rank : int = 512
    qk_nope_head_dim : int = 128
    qk_rope_head_dim : int = 64
    v_head_dim : int = 128

    # yarn
    original_seq_len : int = 4096
    rope_theta : float = 10000.0
    rope_factor : float = 40
    beta_fast : float = 32
    beta_slow : float = 1
    mscale : float = 1.0

    def get_nparams_and_flops(self, model : nn.Module, seq_len : int) -> tuple[int, int]:
        """
        Returns the number of parameters and FLOPs for a given model and sequence length.
        """
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_experts = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.shared_experts" in name:
                nparams_shared_experts += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()


        nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
        nparams = nparams_dense + nparams_sparse

        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_experts
            + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
        )

        logger.info(
            f"Total parameters count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,})"
        )

        n_layers = self.n_layers
        n_heads = self.n_heads
        head_dims = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim

        num_flops_per_token = (
            # 6 * P * N
            6 * (nparams_dense - nparams_embedding + nparams_sparse_active) * n_layers * n_heads * head_dims * seq_len
        )

        return nparams, num_flops_per_token
