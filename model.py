import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# my model location: /home/hzeng/models/Llama-2-7b-hf

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # number of heads for the queries
    n_kv_heads: Optional[int] = 32  # number of heads for the keys and values
    vocab_size: int = -1  # 加载tokenizer时设置
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # kv cache相关
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta:float = 10000.0) -> torch.Tensor:
    # 论文中指出embedding的维度必须是偶数
    assert head_dim % 2 == 0, "head_dim must be even"
    # 计算theta
    # 公式：theta_i = 10000 ^ (-1(i-1)/dim) for i in [1, 2, 3, ..., dim/2]
    # shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape: (head_dim / 2)
    theta = 1.0 / (theta **(theta_numerator / head_dim)).to(device)
    # 计算position
    m = torch.arange(seq_len, device=device)
    # theta和计算position外积相乘
    # shape: (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # 用极坐标表示复数 c = R * exp(i * m * theta), 下面的R = 1
    # shape: (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
    

def apply_rotary_emb(x: torch.Tensor, freqs_complex:torch.Tensor, device: str):
    # 参考https://j5xe3r0637.feishu.cn/docx/VxLldEL6ioCQPix2djLcncWEn2e#share-IlHldoYVRo6UNoxFzLNcXZRwnkb
    # 首先第一步把输入做一些变换，变成复数形式
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (B, seq_len, H, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # 参数：gamma
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # (dim) * (B, seq_len, dim) = (B, seq_len, dim)
        output = self.weight * self._norm(x.float()).type_as(x)
        return output
    
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # 首先引入一个新维度，然后复制，然后reshape
        # (batch_size, seq_len, n_kv_heads, 1, head_dim)
        # (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )
        
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        # kv头数量
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is None else args.n_heads
        # q头数量
        self.n_heads_q = args.n_heads
        # kv头需要重复多少次以匹配q头数量
        self.n_rep = self.n_heads_q // args.n_heads
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads
        
        # qkv矩阵以及kv cache
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim).to(args.device)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim).to(args.device)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (batch_size, 1, dim)
        
        # (batch_size, 1, dim) -> (batch_size, 1, head_q * head_dim)
        xq = self.wq(x)
        # (batch_size, 1, dim) -> (batch_size, 1, head_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (batch_size, 1, head_q * head_dim) -> (batch_size, 1, head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (batch_size, 1, head_kv * head_dim) -> (batch_size, 1, head_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 对q和k应用旋转位置编码(shape没有改变)
        xq = apply_rotary_emb(xq, freqs_complex, device=x.device)
        xk = apply_rotary_emb(xk, freqs_complex, device=x.device)        
        
        # 添加kv cache
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv
        
        # 取出迄今所有的k和v，用于接下来的attention计算
        # (batch_size, seq_len_kv, head_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]
        
        # 复制kv头以匹配q头来计算attention
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # (batch_size, 1, head_q, head_dim) -> (batch_size, head_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        # (batch_size, seq_len_kv, head_q, head_dim) -> (batch_size, head_q, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # (B, head_q, 1, head_dim) @ (B, head_q, head_dim, seq_len_kv) -> (B, head_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # (B, head_q, 1, seq_len_kv) @ (B, head_q, seq_len_kv, head_dim) -> (B, head_q, 1, head_dim)
        output = torch.matmul(scores, values)
        
        # (B, head_q, 1, head_dim) -> (B, 1, head_q, head_dim) -> (B, 1, head_q * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        # 把hidden_dim 四舍五入到最接近 multiple_of 的倍数
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # hidden_size = 7, multiple_of = 5, (7+5-1) // 5 = 2, 2 * 5 = 10
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x)) 
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # 注意力前的Norm
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 前馈神经网络前的Norm
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freq_comlex: torch.Tensor):
        # 简单说就是经过一个注意力，经过一个前馈神经网络，注意都有残差
        # (B, seq_len, dim) + (B, seq_len, dim) = (B, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freq_comlex)        
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, "vocab_size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
            
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_len * 2, device=args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor: 
        # (batch-size, seq-len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"
        
        # (batch-size, seq-len) -> (batch-size, seq-len, dim)
        h = self.tok_embeddings(tokens)
        
        # 
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]
        
        # 经过所有的encoder层
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h)
        return output