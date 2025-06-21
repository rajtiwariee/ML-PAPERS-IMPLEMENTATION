import json
import logging
import math
import time
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import precompute_freqs_cis, apply_rotary_emb
from MOE import MoeArgs, SparseMOE
from config import ModelArgs
from utils import repeat_kv, RMSNorm
import inspect
from TransformerBlock_module import TransformerBlock
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class minimixtral(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.layers = args.n_layers
        self.token_emb = nn.Embedding(args.vocab_size, args.d_model)
        self.layer = nn.ModuleList([TransformerBlock(args=self.args) for _ in range(self.layers)])
        self.norm = RMSNorm(dim=args.d_model)
        self.head = nn.Linear(args.d_model, self.vocab_size)
        self.freqs_complex = precompute_freqs_cis(
            self.args.d_head, self.args.max_seq_len, self.args.device   
        )
        
    def forward(self, x:torch.Tensor,start_pos:int):
        batch_size,seq_len=x.shape
        h=self.token_emb(x)
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]
        for layer in self.layer:
            h=layer(h,freqs_cis=freqs_complex,start_pos=start_pos)
        
        h=self.norm(h)
        out=self.head(h).float()
        
        return out
    
    
    # ───────────────────────────────────────────────────────────────
    def configure_optimizers(self, *, weight_decay: float,
                             learning_rate: float, device: torch.device):
        """
        AdamW with decoupled weight-decay:
            · apply decay to every parameter that has >1 dimension
            · no decay for biases / LayerNorm / RMSNorm weights
            - it applies weight decay to only parameters which have grad enabled
        """
        param_dict = {n: p for n, p in self.named_parameters()
                      if p.requires_grad}

        decay, nodecay = [], []
        for n, p in param_dict.items():
            (decay if p.ndim > 1 else nodecay).append(p)

        optim_groups = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": nodecay,  "weight_decay": 0.0}
        ]
        print(f"▶ decayed tensors:   {len(decay):4d}  "
              f"({sum(p.numel() for p in decay):,} elements)")
        print(f"▶ no-decay tensors:  {len(nodecay):4d}  "
              f"({sum(p.numel() for p in nodecay):,} elements)")

        fused_ok   = ("fused" in inspect.signature(torch.optim.AdamW).parameters
                      and device.type == "cuda")
        print("▶ using fused AdamW:", fused_ok)

        return torch.optim.AdamW(optim_groups,
                                 lr=learning_rate,
                                 betas=(0.9, 0.95),
                                 eps=1e-8,
                                 fused=fused_ok)
        

   
# # ───────────────── WandB ─────────────────
# import wandb


# # train_mixtral_tinystories.py
# import time, math, torch, numpy as np
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from pathlib import Path

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# # ---------- 0. dataset -------------------------------------------------
# from Dataset import NpyShardDataset   # importing dataloader
# # from fineweb import NpyShardDataset
# import os
# # Adjusted hyperparameters for better generalization
# SEQ_LEN = 1024
# BATCH_SIZE = 4  # Reduced batch size
# NUM_WORKERS = os.cpu_count()
# DATA_DIR = Path("tinystories_npy")
# MAX_LR = 1e-4  # Reduced learning rate
# MIN_LR = MAX_LR * 0.05
# WARMUP_STEPS = 50  # Increased warmup
# MAX_STEPS = 5000  # Reduced max steps for early experimentation
# CKPT_DIR = Path("ckpts"); CKPT_DIR.mkdir(exist_ok=True)
# best_val = float("inf")
# VAL_EVERY = 25  # More frequent validation
# GRAD_CLIP = 0.5  # Reduced gradient clipping
# ACCUM_STEPS = 16  # Increased accumulation for stable gradients
# PATIENCE = 5  # Early stopping patience
# NO_IMPROVE_COUNT = 0

# wandb.init(
#     project="mixtral-tinystories",
#     name   ="mixtral_tiny_run_tiny_test_1",
#     save_code=True,
# )

# # log static h-params
# wandb.config.update(dict(seq_len=SEQ_LEN,
#                          batch_size=BATCH_SIZE,
#                          max_lr=MAX_LR,
#                          warmup=WARMUP_STEPS,
#                          max_steps=MAX_STEPS,
#                          weight_decay=0.1))


# device = ("cuda" if torch.cuda.is_available()
#           else "mps" if torch.backends.mps.is_available()
#           else "cpu")
# print("Using", device)

# train_loader = DataLoader(
#     NpyShardDataset(DATA_DIR, split="train", seq_len=SEQ_LEN),
#     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# val_loader   = DataLoader(
#     NpyShardDataset(DATA_DIR, split="val",   seq_len=SEQ_LEN),
#     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True)


# #-------------Increasing the performance with some key changes----------------
# #this sets it from float32 to TF32 -> in which instead of 32 we have 19 bits 
# #making more TFLOps increasing 8 times from float32
# torch.set_float32_matmul_precision('high') #set the precision to high for matmul
# if device == 'cuda':
#     torch.backends.cuda.matmul.allow_tf32 = True
    
    
    



# args  = ModelArgs(vocab_size=50_257, moe=MoeArgs())
# model = minimixtral(args)


# model.to(device)
# model.train()
# wandb.watch(model, log="all", log_freq=10) #record the gradients and computational graph


# # optimizer  = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
# # ▶ cosine LR with linear warm-up

# def lr_at(step: int) -> float:
#     if step < WARMUP_STEPS:                       # linear warm-up
#         return MAX_LR * (step + 1) / WARMUP_STEPS
#     if step >= MAX_STEPS:                         # floor
#         return MIN_LR
#     # cosine decay
#     t = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
#     return MIN_LR + 0.5 * (1 + math.cos(math.pi * t)) * (MAX_LR - MIN_LR)


# # ▶ build the AdamW exactly once, via helper
# optimizer = model.configure_optimizers(weight_decay=0.1,
#                                        learning_rate=MAX_LR,
#                                        device=torch.device(device))
# loss_fn    = torch.nn.CrossEntropyLoss()


# #1. It reads your code in one go and compiles it to a single graph
# #2. It optimizes the graph for performance and improves the speed and read/writes on the gpu
# #3. Read/writes basically so if we perform a simple operation like having a gelu formula( 0.5 * math.pow(2, 0.5) * (x + math.pow(x, 3)))
# #4. so if its normal interpreter it will first copy math.pow(2, 0.5) to the gpu and then perform the operation and then copy it back
# #5. But with torch.compile it will do all the operations in one go and then copy it back to the gpu
# #6. So it will reduce the number of read/writes to the gpu and increase the speed
# if device == 'cuda':
#     model = torch.compile(model) #faster training of the model
# elif device == 'mps':
#     model = torch.compile(model, backend="aot_eager")   # slower but rarely crashes



# train_iter = iter(train_loader)
# val_iter   = iter(val_loader)
# CHECK = True

# for step in range(MAX_STEPS):
#     t0 = time.time()
#     total_loss = 0.0
    
#     # Training phase with label smoothing
#     model.train()
#     for _ in range(ACCUM_STEPS):
#         try:
#             x, y = next(train_iter)
#         except StopIteration:
#             train_iter = iter(train_loader)
#             x, y = next(train_iter)
        
#         x, y = x.to(device), y.to(device)
        
#         logits = model(x, start_pos=0)
        
#         # Label smoothing for regularization
#         smoothing = 0.1
#         confidence = 1.0 - smoothing
#         n_classes = logits.size(-1)
#         log_probs = F.log_softmax(logits.view(-1, n_classes), dim=-1)
        
#         # Create smoothed targets
#         true_dist = torch.zeros_like(log_probs)
#         true_dist.fill_(smoothing / (n_classes - 1))
#         true_dist.scatter_(1, y.view(-1).unsqueeze(1), confidence)
        
#         loss = -torch.sum(true_dist * log_probs) / log_probs.size(0)
        
#         (loss / ACCUM_STEPS).backward()
#         total_loss += loss.item()
    
#     # Gradient clipping and optimization
#     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    
#     # Cosine learning rate schedule
#     lr_now = lr_at(step)
#     for g in optimizer.param_groups:
#         g["lr"] = lr_now
    
#     optimizer.step()
#     optimizer.zero_grad(set_to_none=True)
    
#     dt_ms = (time.time() - t0) * 1000
#     tok_s = BATCH_SIZE * SEQ_LEN * ACCUM_STEPS / (dt_ms / 1000)
#     avg_loss = total_loss / ACCUM_STEPS
    
#     print(f"iter:{step:04d} | loss:{avg_loss:.4f} | g_norm:{grad_norm:.2f} "
#           f"| lr:{lr_now:.2e} | {dt_ms:.0f} ms | {tok_s:.0f} tok/s")
    
#     wandb.log({
#         "step": step,
#         "train/loss": avg_loss,
#         "train/grad_norm": grad_norm.item(),
#         "lr": lr_now,
#         "tokens_per_sec": tok_s
#     })
    
#     # Validation with early stopping
#     if (step + 1) % VAL_EVERY == 0:
#         model_eval = model._orig_mod if hasattr(model, "_orig_mod") else model
#         model_eval.eval()
#         val_losses = []
#         NUM_VAL_BATCHES = 50  # More validation batches
        
#         with torch.inference_mode():
#             for _ in range(NUM_VAL_BATCHES):
#                 try:
#                     xv, yv = next(val_iter)
#                 except StopIteration:
#                     val_iter = iter(val_loader)
#                     xv, yv = next(val_iter)
                
#                 xv, yv = xv.to(device), yv.to(device)
#                 logits_v = model_eval(xv, start_pos=0)
#                 vloss = F.cross_entropy(logits_v.view(-1, logits_v.size(-1)),
#                                       yv.view(-1)).item()
#                 val_losses.append(vloss)
        
#         avg_val_loss = sum(val_losses) / len(val_losses)
#         ppl = math.exp(min(avg_val_loss, 10))  # Cap for numerical stability
        
#         print(f"  ↳ val_loss {avg_val_loss:.4f} | ppl {ppl:.2f}")
#         wandb.log({
#             "step": step,
#             "val/loss": avg_val_loss,
#             "val/ppl": ppl
#         })
        
#         # Early stopping logic
#         if avg_val_loss < best_val:
#             best_val = avg_val_loss
#             NO_IMPROVE_COUNT = 0
            
#             ckpt = CKPT_DIR / f"best_step{step}.pt"
#             model_state = model_eval.state_dict()
#             torch.save({
#                 "model": model_state,
#                 "opt": optimizer.state_dict(),
#                 "step": step,
#                 "val": best_val,
#                 "args": args
#             }, ckpt)
#             wandb.save(str(ckpt))
#             print(f"  ✔ saved new best model → {ckpt}")
#         else:
#             NO_IMPROVE_COUNT += 1
#             if NO_IMPROVE_COUNT >= PATIENCE:
#                 print(f"  ⚠ Early stopping after {PATIENCE} validations without improvement")
#                 break
        
#         model.train()

# print("Training completed!")
# wandb.finish()