import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**(i/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 词向量加大
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        return x

if __name__ == '__main__':
    pe = PositionalEncoder(512, 80)
    x = torch.ones([16, 60, 512]) 
    print(pe(x))
