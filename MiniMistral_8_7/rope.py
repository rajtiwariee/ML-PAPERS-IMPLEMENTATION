import torch
from typing import Tuple

#Here we get the rotation degree how much we have to rotate each embedding value and token
# A token has a higer change while with the embedding have a smaller degree change
#Here the rotation is through the pair 
#so we form a pair (x,y) and we rotate them so if we have one token of embedding size 96 -> pair /rotation angle
#will be 48 rotation angles
def precompute_freqs_cis(dim: int, end: int, device:str,theta:float =10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).to(device)
    t = torch.arange(end, device=device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # (Seq, N_Heads, Head_Dim) --> (Seq, N_Heads, Head_Dim // 2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # (Seq, N_Heads, Head_Dim) --> (Seq, N_Heads, Head_Dim // 2)
    freqs_cis = freqs_cis[:, None, :] # (Seq, 1, Head_Dim // 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2) # (Seq, N_Heads, Head_Dim // 2) * (Seq, 1, Head_Dim // 2) --> (Seq, N_Heads, Head_Dim // 2) --> (Seq, N_Heads, Head_Dim)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)