import torch
import torch.nn as nn
from sam.common import LayerNorm2d
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim: int = 256,
                 num_multimask_outputs: int = 3,
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.num_multimask_outputs - num_multimask_outputs
        
        self.iou_token = nn.Embedding(1, embedding_dim=embed_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embedding_dim=embed_dim)

        self.output_upscaling = nn.Sequential([
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        ])

        self.mlp = nn.Module([
            MLP(input_dim= embed_dim, hidden_dim= embed_dim, output_dim= embed_dim // 8, num_layers= 3)
            for i in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(embed_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    
    def forward(self,
                image_embed_dim: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embedding: torch.Tensor,
                dense_prompt_embedding: torch.Tensor,
                multimask_output: bool):
        
        masks, iou_pred = self.prediction_masks(
            image_embed_dim = image_embed_dim,
            image_pe = image_pe,
            sparse_prompt_embedding = sparse_prompt_embedding,
            dense_prompt_embedding = dense_prompt_embedding
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0,1)
        masks = masks[:, mask_slice, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred


    def predict_masks(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor):

        """
            Predicts masks. See 'forward' for more details.
            -----------------------------------------------
            Args:
                image_embedding: encoder image (B,L,C)
                image_pe: positional encoding of image (C,H,W)
                sparse_prompt_embedding: point, box embedding (B, N, C)
                dense_prompt_embedding: Mask embedding (B, C, H, W)
            Return:

        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        b, l, c = src.shape
        _, _, h,w = dense_prompt_embeddings.shape
        src = torch.repeat_interleave(image_embedding, tokens.shape[0], dim=0)
        # element-wise image embedding and mask embedding 
        dense_prompt_embeddings = dense_prompt_embeddings.permute(0, 2, 3, 1).contigous().view(b, -1, c) # B,C,H,W -> B,H,W,C -> B,L,C
        src = src + dense_prompt_embeddings

        # image positional embedding
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]  # [CLS] token. Output token for IOU prediction, (B, 0, embed_dim)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # Output token for masks (B, num_mask_tokens, embed_dim)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src) # upscale image
        hyper_in_list = []
        for i in range(self.num_mask_tokens): # Run each token through its MLP
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape

        # Dot product of the MLP output for each "output token" and the upscaled image
        # each output token represents a mask
        masks = (hyper_in @ upscaled_embedding.view(b, c, -1)) # B,C,L

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x