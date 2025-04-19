import torch
from torch import nn
from ImageEmbedding import PatchEmbedding
from TransformerEncoderBlock import TransformerEncoderBlock
#1. create a ViT class that inherits the nn.Module
class ViT(nn.Module):
  """
  Creates a vistion Transformer architecture with Vit- base hyperparameters
  """
  def __init__(self,
               img_size: int = 224,
               in_channels: int = 3,
               patch_size: int = 16,
               num_transformer_layers: int = 12,
               embedding_dim : int = 768,
               mlp_size : int = 3072,
               num_heads: int = 12,
               attn_dropout: float = 0,
               mlp_dropout : float = 0.1,
               embedding_dropout: float = 0.1,
               num_classes: int = 1000 #default for imagenet
               ):
    super().__init__()

    #3. Make the image size is divisible by the patch size (if image is small it will throw error)
    assert img_size % patch_size == 0 , 'Image size must be divisible by patch size'

    #4. calculate the number of patches
    number_of_patches = (img_size*img_size) // patch_size**2

    #5. Create learnable class embedding
    self.class_embedding = nn.Parameter(data = torch.randn(1,1,embedding_dim))

    #6. Create learnable position embedding
    self.position_embedding = nn.Parameter(data = torch.randn((1, number_of_patches+1, embedding_dim)))

    #7. Create embedding dropout value
    self.embedding_dropout = nn.Dropout(p = embedding_dropout)

    #8. Create patch embedding layer
    self.patch_embedding = PatchEmbedding(in_channels = in_channels,
                                          patch_size = patch_size,
                                          embedding_dim = embedding_dim)


    # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
    # Note: The "*" means "all"
    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

    #10.Create classifier head
    self.classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape = embedding_dim),
        nn.Linear(in_features = embedding_dim,
                  out_features = num_classes),

    )

  def forward(self,x):

    #12. Get batch size
    batch_size = x.shape[0]

    #13. Create class token embedding and expand it to match the batchsize
    class_token = self.class_embedding.expand(batch_size, -1 ,-1) # # "-1" means to infer the dimension (try this line on its own)

    #14. Create patch embedding (eqn 1)
    x = self.patch_embedding(x)

    #15. Concat class embedding and patch embedding (eqn 1)
    x = torch.cat((class_token,x), dim = 1)

    #16. Add position embedding to patch embedding (eqn 1)
    x = self.position_embedding + x

    #17 Run embedding dropout
    x = self.embedding_dropout(x)

    # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
    x = self.transformer_encoder(x)

    # 19. Put 0 index logit through classifier (equation 4)
    x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index
    #here we extract the class token
    return x