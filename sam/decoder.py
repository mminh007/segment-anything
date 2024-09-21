import torch
import torch.nn as nn
from sam.common import PositionEmbedding
from einops import rearrange

class TransformerDecoder(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 depths: int,
                 mlp_dim:int,
                 downsample_rate: int,
                 dropout: float,
                 qk_scale=None,
                 qkv_bias=None,
                 norm_eps=1e-12):
        super().__init__()

        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.depths = depths

        self.layers = nn.ModuleList()

        for i in range(depths):
            self.layers.append(AttnBlock(embed_dim=embed_dim, num_heads=num_heads,
                                         mlp_dim=mlp_dim, qk_scale=qk_scale,
                                         qkv_bias=qkv_bias, downsample_rate=downsample_rate,
                                         dropout=dropout, norm_eps=norm_eps, skip_pe=(i==0)))
            
        self.final_attn_token_to_image = AttnDecoder(embed_dim, downscale_rate=downsample_rate)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(
            self, image_embedding: torch.Tensor, image_pe: torch.Tensor, point_embedding: torch.Tensor):
        """
        Args:
            image_embedding: Shape (B, H*W, embed_dim)
            point_embedding: Shape (B, N, embed_dim) with N_points
            image_pe: positional embedding location. Shape (C, H, W)
        Return:
            queries: the processed point_embedding (B, N, embed_dim)
            keys: the processed image_embedding (B, H*W, embed_dim)
        """
        B,L,C = image_embedding.shape
        # check shape image_pe
        if len(image_pe.shape) != 4:
            image_pe = image_pe.unsqueeze(0) # (C,H,W) -> (1,C,H,W)
        # reshape
        image_pe = image_pe.permute(0,2,3,1).contiguous().view(1, -1, self.embed_dim)

        queries = point_embedding
        keys = image_embedding
        
        for layer in self.layers:
            queries, keys = layer(queries = queries,
                                  keys = keys,
                                  keys_pe=image_pe,
                                  queries_pe=point_embedding)
        
        q = queries + point_embedding
        k = keys + image_pe

        attn = self.final_attn_token_to_image(q = q, k = k, v = k)
        queries = queries + attn
        queries = self.norm(queries)

        return queries, keys
    

class AttnBlock(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int = 2048,
                 qk_scale=None,
                 qkv_bias=None,
                 downsample_rate: int = 2,
                 dropout=0.1,
                 norm_eps=1e-12,
                 skip_pe=None):
        super().__init__()

        self.attn = AttnDecoder(embed_dim=embed_dim, num_heads=num_heads,
                                     qk_scale=qk_scale, qkv_bias=qkv_bias)
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)

        self.attn_token_to_image = AttnDecoder(embed_dim=embed_dim, num_heads=num_heads,
                                     qk_scale=qk_scale, qkv_bias=qkv_bias, downscale_rate=downsample_rate)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        
        self.attn_image_to_token = AttnDecoder(embed_dim=embed_dim, num_heads=num_heads,
                                     qk_scale=qk_scale, qkv_bias=qkv_bias, downscale_rate=downsample_rate)
        self.norm3 = nn.LayerNorm(embed_dim, eps=norm_eps)
        
        self.mlp = MLPBlock(embed_dim, mlp_dim=mlp_dim, dropout=dropout, norm_eps=norm_eps)
        self.norm4 = nn.LayerNorm(embed_dim, eps=norm_eps)

        self.skip_pe = skip_pe

    def forward(
            self, queries: torch.Tensor, keys: torch.Tensor, keys_pe: torch.Tensor, queries_pe: torch.Tensor):
        """
        Compute Attention Block
        -----------------------
        Args:
            query: point embedding (B, N, embed_dim)
            key: image embedding (B, L, embed_dim) 
            key_pe: positional embedding location for keys
            query_pe: positional embedding for queries
        Return:
            queries: torch.Tensor. Shape (B, N, embed_dim)
            keys: torch.Tensor. Shape(B, L, embed_dim)
        """
        # self-attention
        if self.ski_pe:
            queries = self.attn(queries)
            
        else:
            q = queries + queries_pe
            queries = self.attn(q)
        queries = self.norm1(queries)
        
        # cross attention for tokens to image
        q = queries + queries_pe
        k = keys + keys_pe
        cattn = self.attn_token_to_image(q = queries, k = k, v = keys)
        queries = queries + cattn
        queries = self.norm2(queries)

        # mlp
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # cross attetion for image to token
        q = queries + queries_pe
        k = keys + keys_pe
        cattn_out = self.attn_image_to_token(q = k, k = q, v = queries)
        keys = keys + cattn_out
        keys = self.norm4(keys)

        return queries, keys
    

class MLPBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 mlp_dim: int = 2048,
                 dropout = 0.1,
                 norm_eps = 1e-12):
        
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps = norm_eps)
        self.drop = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class AttnDecoder(nn.Module):
    """
    Attention Base for Transformer Decoder
    -------------------------------------
    Args:
        input_size: size of image_patch
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qk_scale=None,
                 qkv_bias=None,
                 downscale_rate: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        head_dim = embed_dim // num_heads

        self.qk_scale = head_dim ** 0.5 if qk_scale == "None" else qk_scale

        self.hidden_dim = embed_dim // downscale_rate
        assert self.hidden_dim % num_heads == 0, "num_heads must divide embedding_dim"

        self.q_proj= nn.Linear(embed_dim, self.hidden_dim, bias= qkv_bias)
        self.k_proj = nn.Linear(embed_dim, self.hidden_dim, bias = qkv_bias)
        self.v_proj = nn.Linear(embed_dim, self.hidden_dim, bias=qkv_bias)
        
        self.proj = nn.Linear(self.hidden_dim, embed_dim, bias = qkv_bias)

    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor):
        """
        Args:
            x: image encoder. Shape(B, H*W, embed_dim)
             or point embedding. Shape(B, N, embed_dim)
        """
        B,_,_ = q.shape

        # B,_,embed_dim -> B, _, hidden_dim
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v) 
        
        # reshape B, L, C -> B, num_head, L, hidden_dim
        q = rearrange(q, " b l (c n) -> b n l c", n = self.num_heads)
        k = rearrange(k, " b l (c n) -> b n l c", n = self.num_heads)
        v = rearrange(v, " b l (c n) -> b n l c", n = self.num_heads)

        # dot_product(q,k)

        attn = (q * self.qk_scale) @ k.transpose(-2,-1)
        attn = attn / torch.sqrt(q.shape[0])
        attn = torch.softmax(attn, dim = -1)

        out = attn @ v 
        out = self.proj(out) # B * num_head, L, C

        out = out.permute(0, 2, 1, 3).contigous().view(B, -1, self.hidden_dim)
        out = self.proj(out) # B, _, hidden_dim -> B, _, embed_dim
        
        return out
        
