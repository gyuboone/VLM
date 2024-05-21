import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(self,embed_dim=512,n_head=8,USE_BIAS=False,dropout_p=0.):
        super().__init__()
        assert embed_dim%n_head == 0

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.USE_BIAS = USE_BIAS

        self.dropout = nn.Dropout(p=dropout_p)
        
        self.W_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=self.USE_BIAS)   ## single head에서 W_Q(각각의 W_Q)의 차원은 embed_dim * embed_dim//n_head가 되도록 함.
        self.W_K = nn.Linear(self.embed_dim, self.embed_dim, bias=self.USE_BIAS)   ## 여기서는 각각의 W_Q를 만들어서 embedding matrix에 곱하고 concat하는 대신에 한 번에
        self.W_V = nn.Linear(self.embed_dim, self.embed_dim, bias=self.USE_BIAS)   ## embed_dim * (embed_dim//n_head*n_head)가 되도록 matrix를 만듦

        self.W_O = nn.Linear(self.embed_dim, self.embed_dim, bias=self.USE_BIAS)

    def forward(self,x,mask = None):
        ## (batch, seq_size, embed_dim) -> (batch, seq_size, n_head, embed_dim//n_head (논문에서의 d_k)) -> (batch, n_head, seq_size, embed_dim//n_head)
        Q = self.W_Q(x).view(x.shape[0], -1, self.n_head, self.embed_dim//self.n_head).transpose(1,2)  # .permute(0,2,1,3) 과 동일
        K = self.W_K(x).view(x.shape[0], -1, self.n_head, self.embed_dim//self.n_head).transpose(1,2)
        V = self.W_K(x).view(x.shape[0], -1, self.n_head, self.embed_dim//self.n_head).transpose(1,2)

        ## 중요!
        # Q.shape = (n_batches, n_head, seq_size, embed_dim//n_head)
        # K.transpose(-2,-1).shape = (n_batches, n_head, embed_dim//n_head, seq_size)
        # matmul 하게 되면 앞 두차원 n_batches, n_head가 같으므로 1번 head의 query 와 key, 2번 head의 query 와 key .. 끼리만 연산이 된다.
        # Q@K.T 가 multi head attention 내의 모든 query matrix와 key matrix를 곱하는 것이 아님에 주의

        scores = torch.matmul(Q, K.transpose(-2,-1))/Q.size(-1)**0.5
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)

        x = torch.matmul(self.dropout(F.softmax(scores, dim = -1)), V)         # (n_batches, n_head, seq_size, embed_dim//n_head)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.embed_dim) # (n_batches, seq_size, embed_dim)
        return self.W_O(x)



class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class QuickGELU(nn.Module):
    def forward(self, x):
        # See : https://github.com/hendrycks/GELUs
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_head, attn_mask: torch.Tensor = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head

        self.mha = MultiheadAttention(embed_dim=embed_dim, n_head=n_head)
        self.ln1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Linear(embed_dim * 4, embed_dim))
        self.ln2 = LayerNorm(embed_dim)
        self.attn_mask = attn_mask

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))  # Attention is all U need의 transformer와 순서가 약간 다름
        return x



class TextTransformer(nn.Module):
    def __init__(self, embed_dim, n_head, layers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.layer = nn.Sequential(*[ResidualAttentionBlock(embed_dim=embed_dim, n_head=n_head, attn_mask=attn_mask) for _ in range(layers)])

    def foward(self, x):
        return self.layer(x)
    

#####################  text encoder  ########################

