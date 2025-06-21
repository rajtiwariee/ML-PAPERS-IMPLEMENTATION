import os
import math
import torch
import wandb

from model import minimixtral
from config import ModelArgs,MoeArgs
from Dataset import vocab_size,tokenizer,train_dataset,val_dataset,train_loader,val_loader,batch_size
from tqdm import tqdm
import argparse
import torch.nn.functional as F
print(len(val_loader))

device="cuda" if torch.cuda.is_available() else "cpu"
# config = ModelArgs(vocab_size=vocab_size,d_model=512,d_head=64,n_heads=8,n_kv_heads=2,window_size=257,n_experts=8,
#                  top_k=2,n_layers=8,batch_size=batch_size,train_epochs=2,val_epochs=2,seq_len=150,max_seq_len=256,
#                  clip=1,attn_dropout=0.1,dropout=0.1,max_lr=1e-3,beta1=0.9,beta2=0.999,device=device,wandb_project="mixtral",norm_eps=1e-6,attn_eps=1e-6,ffn_eps=1e-6)
# model = tiny_mixtral(args)

config = ModelArgs(
    vocab_size= vocab_size,
    d_model= 512, #embedding size
    d_head= 64, #head size
    n_heads=8, #number of heads
    n_kv_heads=2, #number of key-value heads
    n_layers=8, #number of layers
    train_epochs=2, #number of epochs
    batch_size=8, #batch size
    val_epochs=2, #number of validation epochs
    window_size=257, #window size
    seq_len=256, #sequence length
    max_seq_len=512, #maximum sequence length
    clip=1, #gradient clipping
    attn_dropout=0.1, #attention dropout
    dropout=0.1, #dropout
    max_lr=1e-3, #maximum learning rate
    beta1=0.9, #beta1
    beta2=0.999, #beta2
    n_experts=8, #number of experts
    top_k=2, #top k
    device=device,
    wandb_project='mixtral',
    norm_eps=1e-6,
    attn_eps=1e-6,
    ffn_eps=1e-6,
    moe= MoeArgs())


# In generate_text.py

def load_model(checkpoint_path):
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # --- START OF FIX ---
    # Get the state dict from the checkpoint
    saved_state_dict = checkpoint["model_state_dict"]
    
    # Create a new state dict to hold the cleaned keys
    cleaned_state_dict = {}
    
    # Check if keys have the '_orig_mod.' prefix and remove it
    for key, value in saved_state_dict.items():
        if key.startswith("_orig_mod."):
            # Create a new key without the prefix
            new_key = key[len("_orig_mod."):]
            cleaned_state_dict[new_key] = value
        else:
            # If no prefix, keep the key as is
            cleaned_state_dict[key] = value
            
    # --- END OF FIX ---

    # Make sure you are using the same config as during training!
    # This part is crucial. Let's assume 'config' is globally defined and correct.
    model = minimixtral(args=config).to(device)
    
    # Load the cleaned state dict
    model.load_state_dict(cleaned_state_dict)
    
    model.eval()
    return model

def sample_next_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety
        values, _ = torch.topk(probs, top_k)
        probs[probs < values[..., -1, None]] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

    return torch.multinomial(probs, num_samples=1)


import torch
import torch.nn.functional as F
from tqdm import tqdm # For a nice progress bar

def generate_text(prompt, model, tokenizer, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95):
    """
    Generates text from a trained model given a prompt.
    """
    if not prompt.strip():
        raise ValueError("Prompt is empty. Please provide a non-empty prompt.")

    # 1. Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if input_ids.shape[1] == 0:
        raise ValueError(f"Prompt '{prompt}' produced 0 tokens after tokenization.")

    # 2. Prepare for generation
    output_ids = input_ids.clone()
    model.eval()  # Set the model to evaluation mode

    # 3. Generation loop
    print("\n[INFO] Generating new tokens...")
    with torch.no_grad():
        # Use tqdm for a progress bar
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # --- THE MAIN FIX IS HERE ---
            # Since we are not using a KV cache and are passing the full sequence of
            # generated tokens each time, the start_pos for RoPE should always be 0.
            outputs = model(output_ids, start_pos=0)

            # Get the logits for the very last token in the sequence
            next_token_logits = outputs[:, -1, :]  # Shape: [1, vocab_size]

            # 4. Sample the next token
            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # 5. Check for the End-of-Sentence token
            if next_token_id.item() == tokenizer.eos_token_id:
                print(f"\n[INFO] End-of-sentence token detected. Stopping generation.")
                break

            # 6. Append the new token and continue the loop
            output_ids = torch.cat([output_ids, next_token_id], dim=1)

    # 7. Decode the final sequence of tokens back to a string
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="models/best_epoch.pt", help="Path to model checkpoint")
parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate text from")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
args = parser.parse_args()

model = load_model(args.checkpoint)
result = generate_text(
    args.prompt,
    model,
    tokenizer=tokenizer,
    max_new_tokens=args.max_new_tokens,
    temperature=1.3,   
    top_k=20,
    top_p=0.95
)

print("\n📝 Generated Text:")
print(result)