import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    

def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    # repeat_intrleave repeats the given dimension like this: x = torch.tensor([1, 2, 3]) --> torch.repeat_interleave(x, repeats=2, dim=0) --> torch.tensor([1, 1, 2, 2, 3, 3])
    # This is used to repeat the keys and values to match the number of query heads (Grouped Query Attention).
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim) # (Seq, N_Heads_KV, Head_Dim) --> (Seq, N_Heads, Head_Dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim) # (Seq, N_Heads_KV, Head_Dim) --> (Seq, N_Heads, Head_Dim)
    return keys, values

