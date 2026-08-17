# 第 5 周：CuTe Layout Algebra

- 原始页面：[Week 5: Layout Algebra](https://distinct-capricorn-c04.notion.site/Week-5-Layout-Algebra-30488315b6b48073acd8f4a3a89b3b39)
- 整理日期：2026-08-17
- 英文版：[Week 5: Layout Algebra](../source/notes/05-layout-algebra.en.md)

## Layout 是什么

CuTe 的核心抽象 `Layout` 把多维坐标映射到一维 index。它由 `(Shape, Stride)` 组成：

- `Shape` 定义合法坐标空间；
- `Stride` 定义各坐标分量对线性 index 的贡献；
- 坐标与 stride 做内积即可得到地址偏移。

对二维坐标 `(i,j)` 和 stride `(s_i,s_j)`：

\[
\operatorname{idx}(i,j)=i s_i+j s_j.
\]

Layout 只描述索引规则；把 layout 与 pointer 或 array 组合后，才形成可访问数据的 `Tensor`。

## 行主序、列主序与 Padding

形状 `(4,8)` 的常见布局包括：

- 列主序 `(4,8):(1,4)`，第 0 维连续；
- 行主序 `(4,8):(8,1)`，第 1 维连续；
- 带 padding 的布局，通过让 stride 大于紧凑布局所需跨度，在 index 空间留出空洞。

因此 layout 不等于逻辑 shape。相同 shape 可以对应不同物理组织，也可以映射到非紧凑甚至
交错的地址空间。

## 层次 Shape 与 Stride

CuTe 允许 shape 和 stride 嵌套。例如：

```text
Shape  = (4, (4, 2))
Stride = (4, (1, 16))
```

坐标也相应写成 `(c0,(c1,c2))`，映射为：

\[
\operatorname{idx}(c_0,(c_1,c_2))=4c_0+c_1+16c_2.
\]

嵌套维度可以表达 interleave、warp/lane 分解、tile 内坐标与 tile 坐标等结构，而不必先把
所有层次手工展平成一个难以维护的索引公式。

## Layout Algebra 的意义

普通代数为数定义加、乘、除；layout algebra 则为布局对象定义组合、分解和除法等操作。
它关心的不是数值运算，而是一个坐标空间怎样重解释、分块或映射到另一个索引空间。

常见目标包括：

- 把三维 tensor 折叠为二维矩阵视图；
- 把一个大 layout 分为 tile 内坐标和 tile 网格坐标；
- 组合 thread layout 与 data layout，得到每个线程负责的数据子集；
- 在不改变底层 pointer 的情况下切换逻辑视图。

## 静态值与模板

CuTe 用 `Int<2>{}` 之类的类型把数值声明为编译期常量。模板元编程可让 layout、tile 和
指令选择在编译期特化，从而支持不同 GPU 架构的 MMA 路径，例如 A100 的 `mma.sync`、
H100 的 `wgmma.async` 和 Blackwell 的新一代 Tensor Core 指令。

一个默认二维 layout 示例：

```cpp
auto shape = make_shape(Int<2>{}, Int<4>{});
auto layout = make_layout(shape);
```

默认结果 `(_2,_4):(_1,_2)` 表示 shape `(2,4)` 和列主序 stride `(1,2)`。
调用 `layout(m,n)` 即可把二维坐标映射到一维 index。

## 原练习要求

阅读 CuTe layout 表示与代数，并使用 CuTe 重新实现此前的 FlashAttention-2 算法 1。
练习重点是把 Q/K/V/O 的逻辑 shape、内存 stride、tile 和线程 ownership 写成显式 layout，
而不是立即追求最复杂的 Tensor Core kernel。

## 延伸阅读

- [*CuTe Layout Representation and Algebra*](https://arxiv.org/pdf/2603.02298)；
- [CuTe Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html)，
  建议先读第 0-4 节。

