import numpy as np
import torch
import torch.nn as nn
from sam.common import PositionEmbedding, LayerNorm2d


class PromptEncoder(nn.Module):
    """
    Encode prompts for inputs to SAM's mask decoder
    -------------------------------------------------
    Args:
        embed_dim: prompt_embed_dim: number of input channels
        imgsz: size of image
        input_size: size of patch == imgsz // patch_size
        mask_in_chans: number of hidden channels used for encoding input masks
    """
    def __init__(self,
                 embed_dim: int,
                 imgsz: int,
                 input_size: int,
                 mask_in_chans: int):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.imgsz = imgsz
        self.input_size = input_size
        self.pe_layer = PositionEmbedding(embed_dim // 2)

        self.num_point_embedding = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embedding)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * input_size, 4 * input_size)
        self.mask_downscale = nn.Sequential([
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        ])
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    

    def _embed_points(self, points: torch.Tensor, 
                            labels: torch.Tensor,
                            pad: bool):
        """
            'points': Batch points promts. Shape BxNx2. 
                A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.  
            'labels': Batch labels for point prompts. Shape BxN
                A length N array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a
                background point.
            where N is determined by the number of input points and boxes
        Return:
            point_embedding: torch.Tensor
                pad: shape (B, N + 1, prompt_embed_dim)
                no pad: shape (B, N, prompt_embed_dim)
        """
        points = points + 0.5
        if pad:
            padding_points = torch.zeros((points.shape[0], 1, 2), device = points.device)
            padding_labels = torch.ones((labels.shape[0], 1), device = torch.device)

            points = torch.cat([points, padding_points], dim = 1)
            labels = torch.cat([labels, padding_labels], dim = 1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.imgsz) # positional encoding
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        return point_embedding
    

    def _embed_boxes(self, boxes: torch.Tensor):
        """
        Embeds box prompts:
        ------------------
        Args:
            boxes: Batch of box inputs. Shape Bx4
               A length 4 array given a box prompt to the model. format XYXY
               Example: (32,4) 
        Return:
            corner_embedding: torch.Tensor
                shape: (B, 2, prompt_embed_dim)
        """
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2) # Bx4 -> Bx2x2
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.imgsz) # B,N,prompt_embed_dim
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight

        return corner_embedding


    def _embed_masks(self, masks: torch.Tensor):
        """
        Embed mask inputs
        -----------------
        Args:
            masks: Batched mask inputs to the model. Shape Bx1xHxW
                A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form 1xHxW, where
                for SAM, H=W=256.
        Return:
            masks_embedding: shape (B, prompt_embed_dim, H, W)
        """
        B, C, _, _ = masks.shape 
        masks = masks.permute(0,2,3,1).contiguous().view(B, -1, C) # B,1,H,W -> B,H*W,1
        mask_embedding = self.mask_downscale(masks)
        return mask_embedding

    def _get_batch(self, points, boxes, masks):
        """
        """
        if points is not None:
            return points.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(self,
                points: torch.Tensor,
                boxes: torch.Tensor,
                masks: torch.Tensor):
        """
        Embeds different types of prompts, return both sparse and dense
        ---------------------------------------------------------------
         Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed (B, prompt_embed_dim, H, W)

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes with shape
            (B, N, prompt_embed_dim) (concat points_embedding and boxes embedding) where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(prompt_embed_dim)x(embed_H)x(embed_W)
        """

        batch = self._get_batch(points, boxes, masks)
        sparse_embedding = torch.empty((batch, 0, self.embed_dim), device=self.point_embeddings[0].weight.device)
        
        if points is not None:
            coords, labels = points
            point_embedding = self._embed_points(coords, labels, pad = (boxes is None))
            sparse_embedding = torch.cat([sparse_embedding, point_embedding], dim = 1)
        
        if boxes is not None:
            box_embedding = self._embed_boxes(boxes)
            sparse_embedding = torch.cat([sparse_embedding, box_embedding], dim = 1)
        
        if masks is not None:
            dense_embedding = self._embed_masks(masks)
            dense_embedding = dense_embedding.permute(0, 2, 1).contiguous().view(batch, -1, self.input_size, self.input_size) # B, H*W, prompt_embed_dim -> (B, prompt_embed_dim, input_size, input_size) 
        else:
            dense_embedding = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch, -1, self.input_size, self.input_size) # (B, 1, prompt_embed_dim) -> reshape (1, prompt_embed_dim, 1, 1) -> (bs, -1, input_size, input_size)
            
        return sparse_embedding, dense_embedding


