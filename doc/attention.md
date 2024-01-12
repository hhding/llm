Multi Head Attention:
Q (T, C), K (T, C), V (T, C)
Q (T, h, dk), K (T, h, dk), V (T, h, dk)
Q (h, T, dk), K (h, T, dk), V (h, T, dk)
Q @ K^T: (h, T, dk) @ (h, dk, T) => (h, T, T)

Multi Query Attention:
Q (T, C), K (T, dk), V (T, dk)
Q (T, h, dk), K (T, dk), V (T, dk)
Q @ K^T: (T, C) @ (dk, T) => (T, T)

在计算中，会将 K 扩充，变成跟 Q 同样的维度，然后再进行计算。

Grouped Query Attention:
跟上面不一样的是，K，V 不是只有一组，而是有 N 组独立参数。
然后 Q 也分 N 组，分别和 N 其中一个一个组进行计算。论文里面有一个图，非常清楚。
https://arxiv.org/pdf/2305.13245.pdf
GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
