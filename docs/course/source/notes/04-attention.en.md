# Week 4: Attention, Online Softmax, and FlashAttention

- Original page: [Week 4: Attention](https://distinct-capricorn-c04.notion.site/Week-4-Attention-26a88315b6b480f1b26ffba505ff0677)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 4 notes](../../notes/04-attention.md)

## Weekly Focus

Week 4 extends GEMM tiling to attention and uses online softmax to maintain an
exact normalization state while processing key/value tiles. The course begins
with a correct implementation of FlashAttention-2 Algorithm 1 and defers tensor
cores, warp work partitioning, and lower-level tuning.

## Attention Definition

For `Q,K,V in R^(N x d)`,

\[
O = \operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V.
\]

A naive implementation materializes an `N x N` score or probability matrix in
HBM. FlashAttention changes the evaluation order while preserving exact
semantics, so that intermediate matrix does not need to be written to HBM.

## Query Tiles Define Output Ownership

Split the sequence into query tiles `Q_i` and key/value tiles `K_j,V_j`:

- one thread block owns one `Q_i` and its corresponding output `O_i`;
- the block iterates through every `K_j,V_j` pair;
- each block writes only its own output rows;
- blocks reread key/value tiles but do not race on output rows.

The terms chunk, block, and tile all describe partitioning in the source, but a
CUDA thread block and a mathematical tile are not interchangeable in every
context.

## Shared-Memory Constraint

With four-byte elements, basic storage for `Q_i,O_i,K_j,V_j` is approximately

\[
2B_r d \cdot 4 + 2B_c d \cdot 4.
\]

The algorithm also needs score and probability tiles plus row maxima and
normalizers. The source's complete tile structure yields the element-budget
constraint

\[
M < 2B_c d + 2B_r d + 6B_r + 2B_rB_c.
\]

An implementation must additionally account for registers, alignment, dynamic
shared-memory limits, and any extra state.

## Online Softmax State

For each row, tiled softmax maintains:

- `m`, the maximum over tiles processed so far;
- `l`, the exponential sum relative to `m`;
- `O`, the output accumulator associated with that normalization state.

When a new tile raises the maximum to `m_new`, the old state is rescaled by
`exp(m_old - m_new)` before combining it with the new tile. This is
mathematically equivalent to one-shot softmax without storing the full score
row.

## Meaning of `diag`

`diag(x)` creates a diagonal matrix, so `diag(x)Y` scales row `i` of `Y` by
`x[i]`. `diag(x)^(-1)Y` divides each row by the corresponding value. The paper
uses compact matrix notation; an implementation should not construct the
diagonal matrix.

## Source Exercise

1. implement Algorithm 1 from FlashAttention-2 Section 3.1 in sequential C;
2. implement the same algorithm in CUDA;
3. prioritize correctness before tensor cores, coalescing, or complex warp work
   partitioning;
4. compare against ordinary attention across non-integral tiles, sequence
   lengths, and numerical ranges.

## Further Reading

- [*Online Normalizer Calculation for Softmax*](https://arxiv.org/pdf/1805.02867);
- [*FlashAttention*](https://arxiv.org/pdf/2205.14135), especially pages 1-5;
- [*FlashAttention-2*](https://arxiv.org/pdf/2307.08691), especially pages 1-6;
- [How to Read a Paper](https://6826.csail.mit.edu/2020/papers/howtoread.pdf);
- [How to Read an Engineering Research Paper](https://cseweb.ucsd.edu/~wgg/CSE210/howtoread.html);
- [Online Softmax](../../../chapters/01-online-softmax.md);
- [FlashAttention](../../../chapters/02-flash-attention.md).
