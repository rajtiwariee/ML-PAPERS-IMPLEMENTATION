#gpt2_124M model implementation
#---------Here you can read the paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf-----------


from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

#--------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #key , query , value_projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #torch.tril constructs a lower triangular part
        #which sets upper part to 0

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B, T,C = x.size()

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim = 2)
        
        #(Batch_size, num_heads, Seq_len, Head_dimension)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        
        
        # #attention (materializes the large (T, T)) matrix for all the queries and keys
        # #do the dot matrix multiplication with key and query and multiply with the sqrt to make it stable
        # attention = torch.matmul(q, k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # #here masked_fill tries (to replace the values where the boolean mask with the value)
        # #so if condition becomes True (means 0 is present and need to be masked) we replace it with (-inf)
        # attention = attention.masked_fill(self.bias[:,:,:T,:T] == 0 , float('-inf'))
        # attention = F.softmax(attention, dim = -1)
        # #(Batch_size, num_heads, seq, seq) x (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM) 
        # #(BATCH_SIZE, NUM_HEADS, SEQ, HEAD_DIM)
        # y = attention @ v 
        
        y = F.scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            is_causal = True,
        ) # flash-attention implementation
        
        
        #re-assemble all head outputs side by side
        y = y.transpose(1,2).contiguous().view(B,T,C)

        #output projection
        y = self.c_proj(y)
        return y


#feed forward Linear layers
class MLP(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        #here we expand and then again reduce to make it learn more 
        # (Embed_size, expand_size) = (768, 4*768)
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #bit better then relu
        # (expand_size,Embed_size) = (4*768, 768)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        #residual connection normalised_layer -> self.mlp + x
        x = x + self.mlp(self.ln_2(x)) 
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #sequence length
    vocab_size: int = 50257  #Number of tokens 50,000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    n_layer: int = 12 #Number of transformer blocks
    n_head: int = 4 #Number of attention heads
    n_embd: int = 768 #Embedding size
    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        
        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight #share the weights of the token embedding and the output layer
        
        #we use nn.Module function
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2* self.config.n_layer) ** -0.5 #we are scaling on the number of layers
                #This will control the variance of the weights
                #its 2 times because of attention and mlp
                #with each layer we have sqrt(n) times of std added
                #if we wanna scale we divide it by 1/sqrt(n) this makes it 1
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)   
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.zeros_(module.weight) #not used in gpt2

    def forward(self,idx, targets = None):
        #idx = (batch_size, sequence_length)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        #forward the token and the position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape (T,)
        pos_emb = self.transformer.wpe(pos) #positional embedding of shape   # (B, T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding of shape (T, n_embd) = (1024, 768)
        
        x = tok_emb + pos_emb # (B, T, n_embd)
         
        #forward the block to the transformer (attn and mlp)
        for block in self.transformer.h:
            x = block(x)
            
        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits , loss
    
    def configure_optimizers(self,weight_decay, learning_rate, device):
        #start with all of the candidate parameters
        # Apply weight decay to all parameters except biases and LayerNorm weights
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        #create optim groups Any parameters that is 2D will be weight decays 
        #i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms do not decay
        decay_params = [p for n, p in param_dict.items() if len(p.shape) > 1]
        nodecay_params = [p for n, p in param_dict.items() if len(p.shape) == 1]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parmeter tensors: {len(decay_params)}, with {num_decay_params} total elements")
        print(f"Num nodecayed parmeter tensors: {len(nodecay_params)}, with {num_nodecay_params} total elements")
        
        #create AdamW optimizer and use the fused version if available
        #fused helps to fuse all the kernels into single one kernel
        #this will help to reduce the number of read/writes to the GPU and the operation will be faster
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.type == 'cuda'
        print(f'Using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps = 1e-8,fused=use_fused)
        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type, override_args = None):
        """
        Load a pre-trained model weights from huggingface
        """
        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" %model_type)
        
        #n_layer, n_head and n_embd are determined by the model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),#124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),#350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),#774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),#1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 #always 50257 for gpt model checkpoints
        config_args['block_size'] = 1024
        # config_args['bias'] = True #bias is always true for gpt model checkpoints
        
        # if 'dropout' in override_args:
        #     print(f"Overriding dropout to {override_args['dropout']}")
        #     config_args['dropout'] = override_args['dropout']
        #create a from_scaratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask/ buffer
        
        #init a huggingface / transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        #copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()  
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignore the mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #ignore the mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys) , f" mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the conv1d weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                    
            else:
                #vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
             
        return model
    
