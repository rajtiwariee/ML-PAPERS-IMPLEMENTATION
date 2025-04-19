"This is the implementation of multihead-self-attention , mlp block and transformer encoder block"


import torch
from torch import nn


#1. create a class that inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
  """Creates a multi-head self-attention block ("MSA block" for short)
  """

  #2. Initialize the class with hyperparameters
  def __init__(self, embedding_dim: int = 768,
               num_heads:int = 12,
               attn_dropout: float = 0):
    super().__init__()

    #3. Create a layer normalization
    self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim) #this focuses on that last dimension to normalize it

    #4. Create the Multi-Head Attention (MSA) layer
    self.multihead_attn = nn.MultiheadAttention(embed_dim = embedding_dim,
                                                num_heads = num_heads,
                                                dropout = attn_dropout,
                                                batch_first = True)# does our batch dimension come first?

  #5. Create a forward() method to pass the data through the layers
  def forward(self,X: torch.Tensor) -> torch.Tensor:
    x = self.layer_norm(X)
    attn_output, _ = self.multihead_attn(query = x, #query embeddings
                                         key = x, #key embeddings
                                         value = x ,# value embeddings
                                         need_weights = False )#do we need weights\
    return attn_output




#1. Create a class called MLPBlock that inherits from torch.nn.Module
class MLPBlock(nn.Module):
  "Create the MLPblock with linear and gelu layers"
  def __init__(self,
               embedding_dim:int = 768,
               mlp_size:int = 3072,
               dropout:float = 0.1):
    super().__init__()

    #3. Create a layer Normalization layer
    self.layer_norm = nn.LayerNorm(normalized_shape = embedding_dim)

    #4. Create a sequential series of mlp
    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,  # (768 -> 3072)
                      out_features=mlp_size),
        nn.GELU(), # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,  # (3072 -> 768) # needs to take same in_features as out_features of layer above
                  out_features=embedding_dim), # take back to embedding_dim
        nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
    )

  def forward(self, x : torch.Tensor):

      x = self.layer_norm(x)

      return self.mlp(x)




#1. create a class called TransformerEncoderBlock
class TransformerEncoderBlock(nn.Module):
  """
  Creating a Transformer encoder block
  """
  #2. Instantiate the class with hyperparameters
  def __init__(self,
               embedding_dim:int = 768,
               num_heads: int = 12,
               mlp_size:int = 3072,
               mlp_dropout: float = 0.1, #amount of dropout for dense layers
               attn_dropout: float = 0): #amount of dropout for attention layers
      super().__init__()

      #3.Instantiate a MSA block for using our MultiselfattentionBLock
      self.msa_block = MultiheadSelfAttentionBlock(embedding_dim = embedding_dim,
                                                   num_heads = num_heads)
      #4. Create MLP block
      self.mlp_block = MLPBlock(embedding_dim = embedding_dim,
                                mlp_size = mlp_size,
                                dropout = mlp_dropout)

  def forward(self,x: torch.Tensor):
    # 6. Create residual connection for MSA block (add the input to the output)
      x =  self.msa_block(x) + x
      # 7. Create residual connection for MLP block (add the input to the output)
      x = self.mlp_block(x) + x

      return x