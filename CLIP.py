import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Union


#####################  text encoder start  ########################

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

        # self.mha = MultiheadAttention(embed_dim=embed_dim, n_head=n_head)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_head)
        self.ln1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Linear(embed_dim * 4, embed_dim))
        self.ln2 = LayerNorm(embed_dim)
        self.attn_mask = attn_mask


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.mha(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        x += self.mha(self.ln1(x))
        x += self.mlp(self.ln2(x))  # Attention is all U need의 transformer와 순서가 약간 다름
        return x



class TextTransformer(nn.Module):
    def __init__(self, embed_dim, n_head, layers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.layer = nn.Sequential(*[ResidualAttentionBlock(embed_dim=embed_dim, n_head=n_head, attn_mask=attn_mask) for _ in range(layers)])

    def foward(self, x):
        return self.layer(x)
    
#####################  text encoder end  ########################


##################### image encoder start #######################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(*[
                nn.AvgPool2d(stride),
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            ])

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out



class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)  # spacial_dim : 마지막 feature map 정사각형의 한 변의 길이
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC      H=W=input_resolution//32, N:batch_size, C=embed_dim=width*32
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 공식 구현
        # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x += self.positional_embedding.unsqueeze(1).to(x.dtype)      # (HW+1)NC   text encoder에서는 (seq_size, batch_size, Embed_dim)으로 넣어주는 것과 정확히 대응됨.

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,    # global AvgPool 한 앞 (1,N,C) tensor를 query로 삼음.
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),  # Q,K,V projection에 bias 사용
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)  # ()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # 현재 x의 shape : (B,C,H,W) == (batch_size, width*8, input_resolution//32, input_resolution//32)

        x = self.attnpool(x)

        return x
    

##################### image encoder end #######################


######################## CLIP start ###########################

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )

        self.transformer = TextTransformer(
            embed_dim=transformer_width,
            n_head=transformer_heads,
            layers=transformer_layers,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))       # vision은 이미 embed_dim에 맞게 나오도록 되어 있음

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, seq_size, transformer_width]  NLD

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND             # transformer 안의 nn.MultiheadAttention 에서는 forward할 때 x가 (seq_size, batch_size, transformer_width) 순으로 들어가야 함.
        x = self.transformer(x)                          # nn.Linear 통과할 때는 마지막 demension인 transformer_width에만 적용되므로 transformer 안에서 변환할 필요는 없음
        x = x.permute(1, 0, 2)  # LND -> NLD             # 다시 변환
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection    # (batch_size, seq_size, transformer_width)의 x 중 마지막 토큰의 벡터만 사용 -> (batch_size, transformer_width)
                                                                                       # 여기에 @ (transformer_width, 최종 embed_dim)을 곱하므로 최종 x.shape = (batch_size, embed_dim)이 됨.
                                                                                       # 이로써 encode_img와 차원이 아예 똑같아지고, text와 embedding vector가 일대일 대응됨.
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        ## logit_scale : learned temperature parameter
        logit_scale = self.logit_scale.exp()   ## self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 즉, scalar parameter
        logits_per_image = logit_scale * image_features @ text_features.t()  ## scalar * (batch_size, embed_dim) @ (embed_dim, batch_size)
        logits_per_text = logits_per_image.t()

        # shape = [batch_size, batch_size]
        return logits_per_image, logits_per_text    # 최종 결과물. 두 결과물은 그냥 전치의 차이