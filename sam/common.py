import torch
import torch.nn as nn
import numpy as np


class Patches(nn.Module):
    """
    """
    def __init__(self,
                 kernel_size = 16,
                 stride = 16,
                 padding = None,
                 in_chans = 3,
                 embed_dim = 256,
                 norm_layer = None,
                 eps_norm = 1e-12):
        super().__init()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels = in_chans, out_channels=embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
        if norm_layer:
            self.norm = nn.LayerNorm(normalized_shape=embed_dim, eps = eps_norm)
        

    def forward(self, x):
    
        x = self.proj(x)
        B,_,_,_ = x.shape 

        x = x.permute(0, 2, 3, 1).contigous().view(B, -1, self.embed_dim) # B,embed_dim,H,W -> permute B,H,W,embed_dim -> B, H*W, embed_dim

        if self.norm is None:
            x = self.norm(x)
        
        return x


class PositionEmbedding(nn.Module):
    """
    """
    def __init__(self, num_pos_feat: int, scale: bool):
        super.__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feat)))

    def _pe_encoding(self, coords: torch.Tensor):
        """
        Positionally encode points that are normalized to [0,1]
        -------------------------------------------------------
        Args:
            coords: Shape (B, N, 2) with coords is points 
                or Shape (B, 2, 2) with coords is box
        Return:
            output: Shape(B, N, embed_dim) or (B, 2, embed_dim)
        """
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords

        return torch.cat([torch.sin(coords), torch.cos(coords)], dim = -1)  # B, -1, embed_dim
    

    def forward(self, size: int):
        """
        Generate positional encoding for a grid of the specified size.
        --------------------------------------------------------------
        """
        h, w = size, size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h,w), device = device, dtype = torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5

        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim = -1))

        return pe.permute(2, 0, 1) # C,H,W


    def forward_with_coords(self, coords_input: torch.Tensor, input_size: int):
        """
        Positionally encode points that are not normalized to [0,1].
        ------------------------------------------------------------
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / input_size
        coords[:, :, 1] = coords[:, :, 1] / input_size
        
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
        

class LayerNorm2d(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        """
        LayerNorm along the embed_dim
        -----------------------------
        x: Shape (B, N, embed_dim) or (B, L, embed_dim)
        """

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        #x = self.weight[:, None, None] * x + self.bias[:, None, None]
        x = self.weight[None, :] * x + self.bias[None, :]

        return x
