import torch
import torch.nn as nn
from einops import rearrange




class Attention(nn.Module):
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: int,
                 qk_scale=True,
                 qkv_bias=True,
                 use_rel_pos=True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        head_dim = embed_dim // num_heads
        self.qk_scale = head_dim ** 0.5 if qk_scale == "None" else qk_scale

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size - 1), head_dim)
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size - 1), head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias = qk_scale)

        self.proj = nn.Linear(embed_dim, embed_dim, bias= qk_scale)
    
    def forward(self, x: torch.Tensor, mask = None):
        B, L, C = x.shape
        H, W = self.window_size, self.window_size

        qkv = self.qkv(x) # B, L, C * 3
        q,k,v = tuple(rearrange(qkv, "b l (d f k) -> k (b f) l d", k=3, f=self.num_heads)) # B * num_heads, L, C

        qk_dot_product = torch.einsum("b i d, b j d -> b i j", q, k) * torch.float(self.qk_scale)

        if self.use_rel_pos:
            attn = add_decompose_rel_pos(qk_dot_product, q, self.rel_pos_h, self.rel_pos_w, (H,W), (H,W))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, L, L) + mask.unsqueeze(1)
            attn = attn.view(B, L, L)

            attn = attn.softmax(dim=-1)

        attn = attn.softmax(dim=-1)

        #x = torch.einsum("b i j, b j d -> b i d", attn, v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, L, C)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0,2,3,1,4).reshape(B, L, C)
        x = self.proj(x)

        return x


def window_partition(x, window_size):
    """
        x: (B, H, W, C)
    """

    B,H,W,C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    output = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    
    return output, (Hp, Wp)


def window_unpartition(x, window_size, pad_hw: tuple, hw: tuple):
    """
        x: (B, window_size, window_size, C)
        pad_hw: (Hp, Wp)
        hw: size of input before partition
    """

    h, w = hw
    Hp, Wp = pad_hw

    B = x.shape[0] // (Hp * Wp // window_size // window_size)
    x = x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > h or Wp > w:
        x = x[:, :h, :w, :].contiguous()
    
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor):
    
    """
        q_h: size of query 
        k_h: size of key 
        rel_pos_h: relative positional of height/width
    """ 
    
    max_rel_dist = (2 * max(q_size, k_size) - 1)
    
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = nn.functional.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0,2,1),
                                                    size = max_rel_dist,
                                                    mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1,0)
    
    else:
        rel_pos_resized = rel_pos
    
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decompose_rel_pos(attn: torch.Tensor,
                          q: torch.Tensor,
                          rel_pos_h: torch.Tensor,
                          rel_pos_w: torch.Tensor,
                          q_size: tuple,
                          k_size: tuple):
    """
        attn: attention map q @ k (shape: B* num_heads, H*W, H*W)
        q: query q in the attention layer (B * num_heads, H*W, embed_dim)
        rel_pos_h: relative position embedding for height axis
        rel_pos_w: relative position embedding for width axis
    """

    B,_, dim = q.shape
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    r_q = q.view(B, q_h, q_w).contiguous()
    rel_h= torch.einsum("bhwc, hkc -> bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc, wkc -> bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
            ).view(B, q_h * q_w, k_h * k_w)
    
    return attn