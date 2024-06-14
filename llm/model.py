import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        # d_model = 512
        self.d_model = d_model
        self.h = n_head
        self.d_k = d_model // n_head
        self.scale = math.sqrt(self.d_k)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None, dropout=None):
        # 这里 scale 还是必有，要不然 softmax 产生一些比较极端的输出结果，导致训练不稳定
        # 可以注意一下 softmax 后的分数是不是过于集中或者过于散开，或者训练是否稳定
        # (bs, h, T, d_k) @ (bs, h, d_k, T) => (bs, h, T, T)
        scores = (q @ k.transpose(-1, -2)) / self.scale 
        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)

        # A=(T, T) @ B=(T, d_k)
        # 矩阵A乘以B：A行xB列，因此 softmax 应该操作 A 的行，即 dim=-1
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        # score 的第一行，逐个乘以 v 的列，形成新的第一行，共 d_k 个元素
        # 因此可以认为 score 的每行代表一个头，output 矩阵每一行代表各个头的加权结果
        # (bs, h, T, T) @ (bs, h, T, dk) => (bs, h, T, dk)
        output = scores @ v 
        # 这里是先计算分数，然后 mask，做softmax，dropout，可以换顺序吗？
        # dropout 肯定在最终结果出来后，因此最后
        # softmax 肯定是分数出来之后，mask 影响分数，因此必须在 mask 之后
        # scores 我们可以理解为 T 长度的token互相之间的影响，即 (T, T) 矩阵
        # scores 是通过隐藏的q_linear, k_linear 变换后的 q @ k.transpose 算出来的，这个训练出来的变换的参数才是知识
        # 这里的 q_linear， k_linear 是 (d_model, d_model) 维度的，包括了所有 vocab 之间关系的知识 <- 九阴真经
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0) # q, k, v 的原始结构为 (bs, T, C); size(0)=>bs
        # 矩阵计算只涉及到最后两个维度，即 (T, dk)，效果上相当于变成了按 bs，h 分组了，也就是变成了多头了。
        # 可以看作是将 C 拆成了 h 个 d_k 维矩阵，然后对每个矩阵分别用不同的注意力头去处理。
        # (bs, T, C) => (bs, T, h, d_k) => (bs, h, T, d_k)；这里的 -1 维度是 T。
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.q_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.q_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        z = self.attention(q, k, v, mask, self.dropout) # (bs, h, T, d_k)

        # transpose, slice 等会导致内存不连续，需 contiguous()，才能用 view。
        # (bs, h, T, d_k) => (bs, T, h, d_k) => (bs, T, C)
        concat = z.transpose(1, 2).contiguous().view(bs, -1, self.d_model) 

        return self.out(concat)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        self.ln_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.ln_1(x)))
        x = self.ln_2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        norm = self.alpha * (x - mean)/std + self.bias
        return norm
        

class EncodeBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attn = MultiHeadAttention(n_head, d_model, dropout)
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.dropout_1(self.attn(x))
        x = x + attn_output
        x = self.norm_1(x)
        ff_output = self.dropout_2(self.ff(x))
        x = x + ff_output
        x = self.norm_1(x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, dropout=0.1):
        super().__init__(self)
        self.d_model = d_model
        self.n_head = n_head
        self.pe = PositionEncoder(d_model)
        self.we = nn.Embedding(vocab_size, d_model)
        self.blocks = [EncodeBlock(d_model, n_head, dropout) for _ in range(n_layers)]
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.we(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, x, x, mask)

        return self.norm(x)


class DecodeBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attn_1 = MultiHeadAttention(n_head, d_model, dropout)
        self.attn_2 = MultiHeadAttention(n_head, d_model, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_output, src_mask, trg_mask):
        attn_output_1 = self.dropout_1(self.attn_1(x, x, x, trg_mask))
        x = self.norm_1(x + attn_output_1)
        attn_output_2 = self.dropout_2(self.attn_2(x, e_output, e_output, src_mask))
        x = self.norm_2(x + attn_output_2)
        ff_output = self.dropout_3(self.ff(x))
        x = self.norm_3(x + ff_output)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, dropout=0.1):
        self.d_model = d_model
        self.n_head = n_head
        self.pe = PositionEncoder(d_model)
        self.we = nn.Embedding(vocab_size, d_model)
        self.blocks = [DecodeBlock(d_model, n_head, dropout) for _ in range(n_layers)]
        self.norm = LayerNorm(d_model)

    def forward(self, x, e_output, src_mask, trg_mask):
        x = self.we(x)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, e_output, src_mask, trg_mask)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_head, dropout)
        self.decoder = Decoder(trg_vocab, d_model, n_layers, n_head, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_output = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_output, src_mask, trg_mask)
        output = self.out(d_output)
        return output


if __name__ == '__main__':
    bs=16; T=1024; C=512; h=8
    q = torch.ones([bs, T, C])
    k = torch.ones([bs, T, C])
    c = torch.ones([bs, T, C])
    mha = MultiHeadAttention(h, C)
    mha(q, k, c).shape

