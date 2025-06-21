import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelArgs,MoeArgs
from MOE import MoeArgs,SparseMOE
from Attention import AttentionWithKVCache
from utils import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.args = args
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_head = args.d_head
        
        self.window_size = args.window_size
        self.device = args.device
        self.n_kv_heads=args.n_kv_heads
        self.max_seq_len=args.max_seq_len
        
        
        self.attention=AttentionWithKVCache(dim=self.d_model,num_heads=self.n_heads,window_size=self.window_size
                                            ,device=self.device,max_seq_len=self.max_seq_len,num_kv_heads=self.n_kv_heads,
                                            )
    #     print("Attention dim =", self.attention.dim,
    #   " heads =", self.attention.num_heads,
    #   " head_dim =", self.attention.head_dim)
        self.ffn=SparseMOE(args=self.args)
        
        self.attn_norm=RMSNorm(dim=self.d_model,eps=args.attn_eps)
        self.ffn_norm=RMSNorm(dim=self.d_model,eps=self.args.ffn_eps)
        
    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor, start_pos:int):
        
        r=self.attention(self.attn_norm(x),freqs_cis = freqs_cis, start_pos=start_pos)
        h=x+r #route
        ffn_output, load_balancing_loss=self.ffn.forward(self.ffn_norm(h))

        out=h+ffn_output #route
        return out
        
