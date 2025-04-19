"""This is the implementation of multihead attention from scratch."""
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
  """Multi-headed attention from 'Attention is all you need'"""

  def __init__(self, config):
    self.config = config
    self.embed_dim = config.embed_dim
    self.num_heads = config.num_heads
    self.head_dim = self.embed_dim // self.num_heads
    self.scale = self.head_dim ** -0.5 #Equivalent to 1 / sqrt(self.head_dim)

    self.k_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
    self.q_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
    self.v_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
    self.out_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)

  def forward(self, x : torch.Tensor):
    #[batch_size, num_patches, embed_dim]
    batch_size , seq_len, _ = x.shape

    #query_states: [batch_size, num_patches, embed_dim]
    query_states = self.q_proj(x)
    #key_states : [batch_size,num_patches, embed_dim]
    key_states = self.k_proj(x)
    #value_states : [batch_size, num_patches, embed_dim]
    value_states = self.v_proj(x)
    #split it among the heads

    #[batch_size,num_patches, num_heads,head_dim]
    #1. we split the query between heads [1, 1,8,75] -> 1 seq len is split among 8 heads of 75 dimension
    #2. Then we do the transpose so different heads could be used to see different features of the query
    #3. eg [1,8,4,64] -> so eight heads will be created which will have 4 rows per head of 64 dimension
    #this allows to see different features at a time and later we concat and make it normal
    #providing a broader context of different heads

    query_states = query_states.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1,2)
    key_states = key_states.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1,2)
    value_states = value_states.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1,2)

    #calculate the attn using the formula Q* K^T / sqrt(d_k) , attn_weights: [batch_size, num_heads, seq_len, patches]
    attn_weights = (torch.matmul(query_states,key_states.transpose(2,3))* self.scale)

    if attn_weights.size() != (batch_size, self.num_heads, seq_len,seq_len):
      raise ValueError(
          f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is "
          f"{attn_weights.size()}"
      )

    #apply the softmax row-wise,to make attn_weights into the probabiltiy between 0 and 1 : [batch_size, num_heads, num_patches, num_patches]
    attn_weights = nn.functional.softmax(attn_weights, dim = 1, dtype = torch.float32).to(query_states.dtype)

    #apply dropout only during training
    # attn_weights = nn.functional.dropout(attn_weights, p = self.dropout, training = self.training)

    #multiply the attention weights by the value states(this gives it total context of each values): [batch_size, num_heads, num_patches, head_dim]
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (batch_size,self.num_heads, seq_len, self.head_dim):
      raise ValueError(
          f"attn_output should be of size : {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
          f"{attn_output.size()}"

      )
    #transpose them back
    #[batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads,head_dim]
    attn_output = attn_output.transpose(1,2).contiguous()
    #[batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

    #multiply with Wo (output projection)-we are trying to mix the result
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights
