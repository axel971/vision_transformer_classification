import numpy as np
import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):

        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels = in_channels,
                                 out_channels = embedding_dim,
                                 kernel_size = patch_size,
                                 stride = patch_size,
                                 padding = 0)
        
        self.flatten = nn.Flatten(start_dim = 2,
                                  end_dim = 3)

    def forward(self, x):
        
        # Create an assertion to check that the inputs have the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)

class MultiheadSelfAttentionBlock(nn.Module):
    
    def __init__(self,
                 embedding_dim = 768, # Hidden size D
                 num_heads: int = 12, # Heads
                 attn_dropout: int = 0):
        
        super().__init__()

        # Create the norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)

        # Create multihead self attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim = embedding_dim,
                                                    num_heads = num_heads,
                                                    dropout = attn_dropout,
                                                    batch_first = True) # batch first => (batch, number_of_patches, embedding_dimension)

    def forward(self, x):

        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query = x,
                                             key = x,
                                             value = x,
                                             need_weights = False)
        return attn_output

class MLPBlock(nn.Module):

    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: int = 0.1):

        super().__init__()

        # Create the norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)

        # Create the MLP
        self.mlp = nn.Sequential(
                nn.Linear(in_features = embedding_dim,
                          out_features = mlp_size),
                nn.GELU(),
                nn.Dropout(p = dropout),
                nn.Linear(in_features = mlp_size,
                          out_features = embedding_dim),
                nn.Dropout(p = dropout)
                )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: int = 0.1,
                 attn_dropout: int = 0):

        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim = embedding_dim,
                                                     num_heads = num_heads,
                                                     attn_dropout = attn_dropout)

        # Create MLP block (equation 3)
        self.mlp_block = MLPBlock(embedding_dim = embedding_dim,
                                  mlp_size = mlp_size,
                                  dropout = mlp_dropout)

    def forward(self, x):

        x = self.msa_block(x) + x # residual/skip connection for equation 2
        x = self.mlp_block(x) + x # residual/skip connection for equation 3
        
        return x


# Create a ViT class
class ViT(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: int = 0,
                 mlp_dropout: int = 0.1,
                 embedding_dropout: int = 0.1,
                 num_classes: int = 1000):

        super().__init__()

        # Make an assertion to check if image size is compatible with the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image {img_size} and patch size {patch_size}"

        # Calculate the number of patches (height * width / patch_size**2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels = in_channels,
                                              patch_size = patch_size,
                                              embedding_dim = embedding_dim)

        # Create learnable class embedding
        self.class_embedding = nn.Parameter(data = torch.randn(1, 1, embedding_dim),
                                            requires_grad = True)

        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data = torch.randn(1, self.num_patches + 1, embedding_dim),
                                            requires_grad = True)

        # Create embedding dropout
        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        # Create the Transformer encoder block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim = embedding_dim,
                                                                           num_heads = num_heads,
                                                                           mlp_size = mlp_size,
                                                                           mlp_dropout = mlp_dropout,
                                                                           attn_dropout = attn_dropout) for _ in range(num_transformer_layers)])

        # Create classifier head
        self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape = embedding_dim),
                nn.Linear(in_features = embedding_dim,
                          out_features = num_classes)
                )

    def forward(self, x):

        # Get the batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1) # -1 means infere the dimension

        # Create the class embedding (equation 1)
        x = self.patch_embedding(x)

        # Concatenate class token embedding and patch embedding (eauqtion 1)
        x = torch.cat((class_token, x), dim = 1) # (batch_size, number of patch + 1, embedding dim)

        # Add position embedding to class token and patch embedding (check if position embedding is brodcasted to x dimensions)
        x = self.position_embedding + x

        # Apply dropout to patch embedding 
        x = self.embedding_dropout(x)

        # Pass position and patch embedding to transformer encoder
        x = self.transformer_encoder(x)

        # Puth the 0th index logit throught classier
        x = self.classifier(x[:, 0])

        return x
        
