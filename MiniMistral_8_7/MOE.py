from typing import List
from dataclasses import dataclass
import torch.nn as nn 
import torch
import torch.nn.functional as F
from config import ModelArgs,MoeArgs



class SwiGLUFFN(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int):
        """
        Initializes the SwiGLUFFN module.
        
        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden layer.
            
        Initializes three linear layers:
        - `w_1`: Projects input features to the hidden dimension.
        - `w_2`: Projects input features to the hidden dimension using a separate path.
        - `out`: Projects the transformed hidden representation back to the input dimension.
        """
        super().__init__()
        self.w_1=nn.Linear(input_dim,hidden_dim)
        self.w_2=nn.Linear(input_dim,hidden_dim)
        self.out=nn.Linear(hidden_dim,input_dim)
    def forward(self,x:torch.Tensor):
        """
        Computes the output of the SwiGLUFFN module.
        """
        return self.out(self.w_1(x) * F.silu(self.w_2(x)))


        

class SparseMOE(nn.Module):
    def __init__(self,args:ModelArgs):
        """
        Initializes the SparseMOE module.

        Args:
            d_model (int): The dimensionality of the input features.
            d_hidden (int): The dimensionality of the hidden layer in each expert.
            num_experts (int, optional): The number of expert networks. Defaults to 8.
            top_k (int, optional): The number of experts to be selected for each input. Defaults to 2.

        The module contains a list of expert networks, each an instance of the SwiGLUFFN module,
        and a router to compute the selection distribution over the experts.
        """

        super().__init__()
        self.args = args
        self.d_model=args.d_model
        self.d_hidden=args.hidden_dim
        self.num_experts=self.args.moe.num_experts
        self.top_k=args.top_k
        self.experts=nn.ModuleList([SwiGLUFFN(input_dim=self.d_model,hidden_dim=self.d_hidden) for _ in range(self.num_experts)])
        self.router=nn.Linear(self.d_model,self.num_experts)
    
    def forward(self,x:torch.Tensor):
        """
        Computes the output of the SparseMOE module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,seq_len,d_model).

        Returns:
            tuple: Output tensor of shape (batch_size,seq_len,d_model) and the load balancing loss
        """
        batch_size,seq_len,d_model=x.shape
        
        x_flat=x.view(-1,self.d_model) # (batch_size * seq_len, d_model)
        
        #Step 1: get router scores for each token
        router_logits=self.router(x_flat) # (batch_size * seq_len, num_experts)
        router_probs=F.softmax(router_logits,dim=-1)
        
        #Step 2: get top-k experts
        topk_probs,topk_indices=torch.topk(router_probs,self.top_k,dim=-1) #(batch_size*seq_len, top_k)
        #for token0 : topk_probs : [0.5,0.4] topk_indices : [2,0] -> expert 2 and expert 0
        #Step 3: compute weighted sum of top-k experts
        expert_outputs=[]
        for i in range(self.top_k):
            expert_idx=topk_indices[:,i]
            outputs=torch.zeros_like(x_flat)
            
            for expert_id in range(self.num_experts):
                mask=(expert_id==expert_idx)
                if mask.any():
                    selected_x=x_flat[mask]
                    expert_out=self.experts[expert_id](selected_x)
                    outputs[mask]=expert_out
            
            weighted_output = topk_probs[:, i].unsqueeze(-1) * outputs
            expert_outputs.append(weighted_output)
        
        final_output = sum(expert_outputs)
        
        final_output = final_output.view(batch_size, seq_len, d_model)
        
        router_probs_mean = router_probs.mean(dim=0)
        load_balancing_loss = (router_probs_mean * router_probs_mean).sum() * self.num_experts
        
        return final_output, load_balancing_loss