"""Small CPU demo for the supported executable specifications."""

import torch

from ds_flash_mla_moe import blockwise_attention, deepseek_moe_reference


def main() -> None:
    torch.manual_seed(7)

    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 12)
    attention_output = blockwise_attention(q, k, v, causal=True, block_size=3)

    tokens, model_dim, hidden, experts = 6, 8, 12, 4
    x = torch.randn(tokens, model_dim)
    gate = torch.randn(experts, model_dim)
    w1 = torch.randn(experts, hidden, model_dim)
    w2 = torch.randn(experts, model_dim, hidden)
    w3 = torch.randn(experts, hidden, model_dim)
    moe_output = deepseek_moe_reference(
        x,
        gate,
        w1,
        w2,
        w3,
        topk=2,
        n_groups=2,
        topk_groups=1,
    )

    print("attention:", tuple(attention_output.shape))
    print("moe:", tuple(moe_output.shape))


if __name__ == "__main__":
    main()
