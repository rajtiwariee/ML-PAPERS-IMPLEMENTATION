# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from rope import precompute_freqs_cis, apply_rotary_emb
# from utils import repeat_kv
# import einops
# import math


# class AttentionWithKVCache(nn.Module):
#     def __init__(self, dim: int, num_heads: int, window_size: int, device, max_seq_len: int = 2048, num_kv_heads:int=2):
#         """
#         Initialize the MultiHeadedAttention module with KV cache.

#         Args:
#             dim (int): The dimensionality of the input and output features.
#             num_heads (int): The number of attention heads.
#             window_size (int): The size of the window for rolling buffer KV cache.
#             max_seq_len (int, optional): The maximum sequence length for initialization of KV cache. Defaults to 2048.
#             num_kv_heads (int, optional): The number of attention heads for KV projection. Defaults to 2.
#         """
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.num_kv_heads=num_kv_heads
#         self.max_seq_len = max_seq_len
#         self.repeats=num_heads//num_kv_heads
#         self.window_size=window_size
#         self.half_window=self.window_size//2
#         self.device=device
        
#         # Projection layers
#         self.W_q = nn.Linear(dim,self.num_heads*self.head_dim)
#         self.W_k = nn.Linear(dim, self.num_kv_heads*self.head_dim)
#         self.W_v = nn.Linear(dim, self.num_kv_heads*self.head_dim)
#         self.W_o = nn.Linear(dim, dim)

#         # Initialize KV cache
#         # self.register_buffer('cache_k', torch.zeros(
#         #     (1, max_seq_len, self.num_kv_heads, self.head_dim)  # batch=1 for simplicity
#         # ))
#         # self.register_buffer('cache_v', torch.zeros(
#         #     (1, max_seq_len, self.num_kv_heads, self.head_dim)
#         # ))
#         self.register_buffer('cache_k', torch.zeros((max_seq_len, self.num_kv_heads, self.head_dim)))
#         self.register_buffer('cache_v', torch.zeros((max_seq_len, self.num_kv_heads, self.head_dim)))
        
#         # self.cache_k=None
#         # self.cache_v=None
#         self.cache_pos=0
    
#     def update_cache(self,seq_len:int,k:torch.Tensor,v:torch.Tensor):
#         """
#         Update KV cache with new key-value pairs.

#         Args:
#             seq_len (int): The sequence length of the new key-value pairs.
#             k (torch.Tensor): The new key tensor of shape (batch_size, seq_len, num_heads, head_dim).
#             v (torch.Tensor): The new value tensor of shape (batch_size, seq_len, num_heads, head_dim).

#         Returns:
#             None
#         """
#         seq_len=k.size(1)
        
#         if self.cache_pos + seq_len > self.max_seq_len: #check if cache has enough space
#             #roll the cache to make space
#             roll_amount=seq_len
#             self.cache_k=torch.roll(self.cache_k,shifts=-roll_amount,dims=0)
#             self.cache_v=torch.roll(self.cache_v,shifts=-roll_amount,dims=0)
#             self.cache_pos-=roll_amount
        
#         self.cache_k[self.cache_pos:self.cache_pos+seq_len]=k.squeeze(0)
#         self.cache_v[self.cache_pos:self.cache_pos+seq_len]=v.squeeze(0)
#         self.cache_pos+=seq_len


#     def forward(self, x: torch.Tensor,freqs_cis: torch.Tensor, start_pos: int = 0 ):
#         """
#         Compute attention with KV cache.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
#             start_pos (int, optional): The starting position of the sequence. Defaults to 0.

#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
#         """
#         batch_size, seq_len, _ = x.shape
#         assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds max sequence length {self.max_seq_len}."
        
#         q = self.W_q(x) # (B, Seq, Dim) --> (B, Seq, N_Heads * Head_Dim)
#         k = self.W_k(x) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV * Head_Dim)
#         v = self.W_v(x) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV * Head_Dim)

#         q = q.view(batch_size, seq_len, self.num_heads, self.head_dim) # (B, Seq, Dim) --> (B, Seq, N_Heads, Head_Dim)
#         k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV, Head_Dim)
#         v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV, Head_Dim)
        
        
#         #apply rotatory embedding
#         # Store original shapes before apply_rotary_emb
#         q_shape_before_rope = q.shape
#         k_shape_before_rope = k.shape

#         #apply rotatory embedding
#         if freqs_cis is not None:
#             q,k = apply_rotary_emb(q,k,freqs_cis)
            
#             # If apply_rotary_emb changed the number of dimensions, reshape back
#             if q.ndim != len(q_shape_before_rope):
#                 # print(f"Warning: q.ndim changed from {len(q_shape_before_rope)} to {q.ndim} after apply_rotary_emb. Reshaping back.")
#                 q = q.view(q_shape_before_rope)
#             if k.ndim != len(k_shape_before_rope):
#                 # print(f"Warning: k.ndim changed from {len(k_shape_before_rope)} to {k.ndim} after apply_rotary_emb. Reshaping back.")
#                 k = k.view(k_shape_before_rope)
            
#         if self.training: #this variable is get initialize when you do model.train()
#             #full training mode
#             # q = q.transpose(1,2)
#             # k = k.transpose(1,2)
#             # v = v.transpose(1,2)
#             # print(f'before permute : {q.ndim}')
#             q = q.permute(0, 2, 1, 3)        # q: (B, N_H, S, H_D)
#             k = k.permute(0, 2, 1, 3)        # k: (B, N_KV_H, S, H_D)
#             v = v.permute(0, 2, 1, 3)        # v: (B, N_KV_H, S, H_D)
            
#             # print(f'q and k after permute : {q.shape},{k.shape}')
#             k,v = repeat_kv(k, v, self.repeats, dim=1) # k,v: (B, N_H, S, H_D)
            
#             pad_k = F.pad(k, (0,0,self.half_window,self.half_window))
#             pad_v = F.pad(v, (0,0,self.half_window,self.half_window))
            
#             # k_unf from unfold will be (B, N_H, S_out, H_D, W_size)
#             k_unf = pad_k.unfold(2, self.window_size, step=1) 
#             v_unf = pad_v.unfold(2, self.window_size, step=1) 
            
#             # Prepare q for einsum: (B, N_H, S, 1, H_D)
#             q_e = q.unsqueeze(3)
            
#             # Prepare k_unf for einsum: (B, N_H, S, W_size, H_D)
#             # Current k_unf is (B, N_H, S_out, H_D, W_size)
#             # S_out should be S if window_size is odd (e.g., 5), which it is.
#             # So S_out = S.
#             assert k_unf.shape[2] == q_e.shape[2], f"Seq len mismatch k_unf {k_unf.shape[2]} vs q {q_e.shape[2]}"

#             # Permute H_D and W_size for k_unf
#             # (B, N_H, S, H_D, W_size) -> (B, N_H, S, W_size, H_D)
#             k_e = k_unf.permute(0, 1, 2, 4, 3)
            
#             # Permute H_D and W_size for v_unf
#             v_e = v_unf.permute(0, 1, 2, 4, 3)

#             # Print shapes of tensors going into einsum
#             # print(f'q_e shape : {q_e.shape}')
#             # print(f'k_e shape : {k_e.shape}')
            
#             attn_scores=einops.einsum(q_e, k_e,'b h s w_q d, b h s w_kv d -> b h s w_kv ')
#             # Note: using w_q and w_kv to be explicit. einops 'b h s w d, b h s w d -> b h s w'
#             # would also work due to broadcasting rule (w_q=1).
            
#             mask=torch.tril(torch.ones(self.window_size,self.window_size,device=self.device))
#             mask=mask[-1] # This mask might need review for correctness in sliding window causal attention
#             mask=mask.view(1,1,1,self.window_size).expand(batch_size,self.num_heads,seq_len,self.window_size)
#             attn_scores = attn_scores.masked_fill(mask == 0, float('-inf')) # Make sure mask == 0 is the correct condition
            
#             attn_weights = F.softmax(attn_scores, dim=-1) # dim=-1 is the w_kv dimension
            
#             # output 'b h s d'
#             # attn_weights 'b h s w_kv'
#             # v_e 'b h s w_kv d'
#             output=einops.einsum(attn_weights, v_e,'b h s w, b h s w d -> b h s d')

        
#         else:
#             #batch_size must be 1 for inference
#             assert batch_size==1, "batch size must be 1"
#             # Inference mode - use KV cache
#             # Update cache
#             self.update_cache(seq_len,k,v)
#             current_len=min(self.cache_pos,self.max_seq_len)
#             valid_cache_len=min(current_len,self.window_size)
            
#             start_window=max(0,current_len-seq_len-self.half_window)
                
#             if self.cache_k is None or self.cache_v is None:
#                 self.cache_k=torch.zeros((self.max_seq_len,self.num_kv_heads,self.head_dim))
#                 self.cache_v=torch.zeros((self.max_seq_len,self.num_kv_heads,self.head_dim))
            
           
#             cached_k=self.cache_k[start_window:current_len].unsqueeze(0)
#             cached_v=self.cache_v[start_window:current_len].unsqueeze(0)
            
#             valid_cache_len=cached_k.shape[-2]
            

#             q=q.transpose(1,2)
#             cached_k=cached_k.transpose(1,2)
#             cached_v=cached_v.transpose(1,2)            # Compute attention
#             cached_k=cached_k.repeat_interleave(self.repeats,dim=1)
#             cached_v=cached_v.repeat_interleave(self.repeats,dim=1)
           
#             attn_scores = torch.matmul(q, cached_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
        
#             valid_cache_len = cached_k.shape[-2]

#             mask = torch.ones((seq_len, valid_cache_len), dtype=torch.bool, device=x.device)
#             mask = torch.tril(mask, diagonal=valid_cache_len - seq_len)

#             mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, valid_cache_len)
#             mask = mask.expand(batch_size, self.num_heads, seq_len, valid_cache_len)

#             attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
#             # print("attn_scores shape:", attn_scores.shape)
#             # print("mask shape:", mask.shape)
#             #attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
#             attn_probs = F.softmax(attn_scores, dim=-1)
#             output = torch.matmul(attn_probs, cached_v)

#         # Output projection
#         output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
#         return self.W_o(output)

#     def reset_cache(self):
#         """Reset the KV cache between sequences"""
#         self.cache_k.zero_()
#         self.cache_v.zero_()
#         self.cache_pos = 0
   


#new code
import torch
import torch.nn as nn
from torch.nn import functional as F
from rope import precompute_freqs_cis, apply_rotary_emb
from utils import repeat_kv
import einops
import math


class AttentionWithKVCache(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, device, max_seq_len: int = 2048, num_kv_heads:int=2):
        """
        Initialize the MultiHeadedAttention module with KV cache.

        Args:
            dim (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            window_size (int): The size of the window for rolling buffer KV cache.
            max_seq_len (int, optional): The maximum sequence length for initialization of KV cache. Defaults to 2048.
            num_kv_heads (int, optional): The number of attention heads for KV projection. Defaults to 2.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_kv_heads=num_kv_heads
        self.max_seq_len = max_seq_len
        self.repeats=num_heads//num_kv_heads
        self.window_size=window_size
        self.half_window=self.window_size//2
        self.device=device
        
        # Projection layers
        self.W_q = nn.Linear(dim,self.num_heads*self.head_dim)
        self.W_k = nn.Linear(dim, self.num_kv_heads*self.head_dim)
        self.W_v = nn.Linear(dim, self.num_kv_heads*self.head_dim)
        self.W_o = nn.Linear(dim, dim)

        # Initialize KV cache - now supports multiple batch sizes
        # We'll initialize these dynamically based on the first batch size we see
        self.cache_k = None
        self.cache_v = None
        self.cache_pos = 0
        self.cache_batch_size = None
    
    def _initialize_cache(self, batch_size: int):
        """Initialize cache buffers for the given batch size"""
        if self.cache_k is None or self.cache_batch_size != batch_size:
            self.cache_k = torch.zeros(
                (batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim),
                device=self.device,
                dtype=torch.float32
            )
            self.cache_v = torch.zeros(
                (batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim),
                device=self.device,
                dtype=torch.float32
            )
            self.cache_batch_size = batch_size
            self.cache_pos = 0
    
    def update_cache(self, seq_len: int, k: torch.Tensor, v: torch.Tensor):
        """
        Update KV cache with new key-value pairs.

        Args:
            seq_len (int): The sequence length of the new key-value pairs.
            k (torch.Tensor): The new key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim).
            v (torch.Tensor): The new value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim).

        Returns:
            None
        """
        batch_size = k.size(0)
        seq_len = k.size(1)
        
        # Initialize cache if needed
        self._initialize_cache(batch_size)
        
        if self.cache_pos + seq_len > self.max_seq_len:
            # Roll the cache to make space
            roll_amount = seq_len
            self.cache_k = torch.roll(self.cache_k, shifts=-roll_amount, dims=1)
            self.cache_v = torch.roll(self.cache_v, shifts=-roll_amount, dims=1)
            self.cache_pos -= roll_amount
        
        # Update cache for all batches
        self.cache_k[:, self.cache_pos:self.cache_pos+seq_len] = k
        self.cache_v[:, self.cache_pos:self.cache_pos+seq_len] = v
        self.cache_pos += seq_len

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0):
        """
        Compute attention with KV cache.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Rotary embedding frequencies.
            start_pos (int, optional): The starting position of the sequence. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape
        assert seq_len <= self.max_seq_len, f"Input sequence length {seq_len} exceeds max sequence length {self.max_seq_len}."
        
        q = self.W_q(x) # (B, Seq, Dim) --> (B, Seq, N_Heads * Head_Dim)
        k = self.W_k(x) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV * Head_Dim)
        v = self.W_v(x) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV * Head_Dim)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim) # (B, Seq, Dim) --> (B, Seq, N_Heads, Head_Dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV, Head_Dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # (B, Seq, Dim) --> (B, Seq, N_Heads_KV, Head_Dim)
        
        # Store original shapes before apply_rotary_emb
        q_shape_before_rope = q.shape
        k_shape_before_rope = k.shape

        # Apply rotatory embedding
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
            
            # If apply_rotary_emb changed the number of dimensions, reshape back
            if q.ndim != len(q_shape_before_rope):
                q = q.view(q_shape_before_rope)
            if k.ndim != len(k_shape_before_rope):
                k = k.view(k_shape_before_rope)
            
        if self.training:
            # Full training mode with sliding window
            q = q.permute(0, 2, 1, 3)        # q: (B, N_H, S, H_D)
            k = k.permute(0, 2, 1, 3)        # k: (B, N_KV_H, S, H_D)
            v = v.permute(0, 2, 1, 3)        # v: (B, N_KV_H, S, H_D)
            
            k, v = repeat_kv(k, v, self.repeats, dim=1) # k,v: (B, N_H, S, H_D)
            
            pad_k = F.pad(k, (0,0,self.half_window,self.half_window))
            pad_v = F.pad(v, (0,0,self.half_window,self.half_window))
            
            k_unf = pad_k.unfold(2, self.window_size, step=1) 
            v_unf = pad_v.unfold(2, self.window_size, step=1) 
            
            q_e = q.unsqueeze(3)
            
            assert k_unf.shape[2] == q_e.shape[2], f"Seq len mismatch k_unf {k_unf.shape[2]} vs q {q_e.shape[2]}"

            k_e = k_unf.permute(0, 1, 2, 4, 3)
            v_e = v_unf.permute(0, 1, 2, 4, 3)
            
            attn_scores = einops.einsum(q_e, k_e,'b h s w_q d, b h s w_kv d -> b h s w_kv ')
            
            mask = torch.tril(torch.ones(self.window_size, self.window_size, device=self.device))
            mask = mask[-1]
            mask = mask.view(1,1,1,self.window_size).expand(batch_size, self.num_heads, seq_len, self.window_size)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            output = einops.einsum(attn_weights, v_e,'b h s w, b h s w d -> b h s d')

        else:
            # Inference mode - now supports any batch size
            # Initialize cache if needed
            self._initialize_cache(batch_size)
            
            # Update cache
            self.update_cache(seq_len, k, v)
            current_len = min(self.cache_pos, self.max_seq_len)
            
            # Calculate window boundaries
            start_window = max(0, current_len - seq_len - self.half_window)
            end_window = current_len
            
            # Get cached keys and values for the window
            cached_k = self.cache_k[:, start_window:end_window]  # (B, window_len, num_kv_heads, head_dim)
            cached_v = self.cache_v[:, start_window:end_window]  # (B, window_len, num_kv_heads, head_dim)
            
            valid_cache_len = cached_k.shape[1]
            
            # Reshape for attention computation
            q = q.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
            cached_k = cached_k.permute(0, 2, 1, 3)  # (B, num_kv_heads, window_len, head_dim)
            cached_v = cached_v.permute(0, 2, 1, 3)  # (B, num_kv_heads, window_len, head_dim)
            
            # Repeat KV heads to match query heads
            cached_k = cached_k.repeat_interleave(self.repeats, dim=1)  # (B, num_heads, window_len, head_dim)
            cached_v = cached_v.repeat_interleave(self.repeats, dim=1)  # (B, num_heads, window_len, head_dim)
           
            # Compute attention scores
            attn_scores = torch.matmul(q, cached_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Create causal mask for the window
            mask = torch.ones((seq_len, valid_cache_len), dtype=torch.bool, device=x.device)
            mask = torch.tril(mask, diagonal=valid_cache_len - seq_len)
            
            # Expand mask for all heads and batches
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, valid_cache_len)
            mask = mask.expand(batch_size, self.num_heads, seq_len, valid_cache_len)
            
            # Apply mask
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_probs, cached_v)

        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)

    def reset_cache(self):
        """Reset the KV cache between sequences"""
        if self.cache_k is not None:
            self.cache_k.zero_()
        if self.cache_v is not None:
            self.cache_v.zero_()
        self.cache_pos = 0
        self.cache_batch_size = None