# Week 5: CuTe Layout Algebra

- Original page: [Week 5: Layout Algebra](https://distinct-capricorn-c04.notion.site/Week-5-Layout-Algebra-30488315b6b48073acd8f4a3a89b3b39)
- Snapshot prepared: 2026-08-17
- Chinese version: [Week 5 notes](../../notes/05-layout-algebra.md)

## What Is a Layout?

CuTe's central `Layout` abstraction maps multidimensional coordinates to a
one-dimensional index. A layout is a `(Shape, Stride)` pair:

- `Shape` defines the valid coordinate space;
- `Stride` defines each coordinate component's contribution to the index;
- the coordinate-stride inner product gives the address offset.

For coordinate `(i,j)` and stride `(s_i,s_j)`,

\[
\operatorname{idx}(i,j)=i s_i+j s_j.
\]

A layout describes indexing only. Combining it with a pointer or array produces
a `Tensor` that can access data.

## Major Order and Padding

For shape `(4,8)`:

- `(4,8):(1,4)` is column-major because mode 0 is contiguous;
- `(4,8):(8,1)` is row-major because mode 1 is contiguous;
- larger strides can leave padding holes in index space.

Logical shape therefore does not determine physical organization. The same
shape can represent compact, padded, or interleaved storage.

## Hierarchical Shape and Stride

CuTe permits nested shapes and strides:

```text
Shape  = (4, (4, 2))
Stride = (4, (1, 16))
```

The matching coordinate `(c0,(c1,c2))` maps as

\[
\operatorname{idx}(c_0,(c_1,c_2))=4c_0+c_1+16c_2.
\]

Nested modes can express interleaving, warp/lane decomposition, and tile-internal
versus tile-grid coordinates without flattening every relationship into manual
pointer arithmetic.

## Purpose of Layout Algebra

Layout algebra defines composition, decomposition, and division over layout
objects. It is used to reinterpret or tile coordinate spaces rather than to
perform numerical arithmetic.

Typical uses include folding a 3-D tensor into a matrix view, dividing a large
layout into within-tile and tile-grid coordinates, composing a thread layout
with a data layout, and changing logical views without changing the pointer.

## Static Values and Templates

CuTe types such as `Int<2>{}` make a value a compile-time constant. Template
metaprogramming lets layouts, tiles, and instructions specialize for different
GPU architectures, including `mma.sync`, `wgmma.async`, and newer Blackwell
tensor-core paths.

```cpp
auto shape = make_shape(Int<2>{}, Int<4>{});
auto layout = make_layout(shape);
```

The default result `(_2,_4):(_1,_2)` is a `(2,4)` shape with column-major stride
`(1,2)`. Calling `layout(m,n)` maps the coordinate to an index.

## Source Exercise

Study CuTe layout representation and algebra, then reimplement the earlier
FlashAttention-2 Algorithm 1 with CuTe. The main goal is to express Q/K/V/O
shapes, strides, tiles, and thread ownership explicitly before pursuing the most
advanced tensor-core kernel.

## Further Reading

- [*CuTe Layout Representation and Algebra*](https://arxiv.org/pdf/2603.02298);
- [CuTe Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html),
  especially Sections 0-4.
