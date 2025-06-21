# minimixtral 175M MoE

---

## 🧠 About Tiny-Mixtral

This project is a simplified re-implementation of the **Mixtral of Experts** (MoE) architecture, inspired by the paper *"Mixtral of Experts: Sparse Mixture of Experts for Efficient Language Modeling"*. It aims to reproduce core ideas like sparse routing, expert selection, and caching strategies in a lightweight and educational way.

### ⚙️ Core Concepts

- **Mixture of Experts (MoE):**
  Instead of using all model parameters at each step, only a subset of "experts" (typically 2 out of 8) are activated per token. This significantly reduces compute while keeping performance high.
- **Gated Query Attention (GQA):**
  An optimization of multi-query attention, where each head shares key and value projections, but allows for more nuanced gating of different experts based on the query.
- **Key-Value (KV) Caching:**
  Speeds up autoregressive generation by storing key and value tensors from previous forward passes, avoiding redundant computation during inference.
- **Sliding Window Attention:**
  Replaces full attention with local attention, limiting context to a fixed window size. This improves memory efficiency and runtime, especially for long sequences.
- **Rolling Buffer KV Cache:**
  Implements a memory-efficient rolling cache that discards the oldest tokens as new ones come in, while maintaining relevant recent context for the model.

---

This minimal implementation is suitable for understanding how modern LLM optimizations work, especially in resource-constrained environments or for academic exploration.

---

Example of a generated text:

```
Once upon a time blue came home youo noise down lots it riendsoo thoughto some you you you themeed back you you you you
```

## 📊 Training Insights

### 📁 Dataset

Training was done using the [TinyStories](https://huggingface.co/datasets/tiny_stories) dataset available on Hugging Face.

### ⚙️ Hardware

- **GPU Used:** NVIDIA L4

### 📈 First 2 Epochs: Learning Rate & Training Loss

The plot below shows how the training learning rate and loss behaved during the first two epochs:

![Training Curve](static/first_2_epochs.png)

## 🔧 Setup

```bash
# Clone the repository
git clone https://github.com/rajtiwariee/repo
cd mini-mixtral
# Install dependencies
pip3 install -r requirements.txt
# Log in to Hugging Face
huggingface-cli login
# Log in to Weights & Biases (W&B)
import wandb
wandb.login()
#train
python train.py --usewandb
#resume training from a checkpoint
python train.py --usewandb --checkpoint models/best_epoch.pt
#generate
python generate_text.py --prompt "Once upon a time" --max_new_tokens 20

```

---

### Parameter Calculation for MoE Transformer (Mixtral-like)

### 📌 Model Hyperparameters

| Parameter      | Value |
| -------------- | ----- |
| `vocab_size` | 32000 |
| `d_model`    | 512   |
| `d_head`     | 64    |
| `n_heads`    | 8     |
| `n_kv_heads` | 2     |
| `n_experts`  | 8     |
| `top_k`      | 2     |
| `n_layers`   | 8     |

---

#### 🧠 1. Token Embedding

Each token gets a `d_model`-dim vector.
Embedding=vocab_size x d_model = 32000 x 512 = 16,384,000

---

#### 🧠 2. Attention Layer (per layer)

#### a. QKV Projections

- **Query**: `512 × 512 = 262,144`
- **Key**: `512 × 128 = 65,536`
- **Value**: `512 × 128 = 65,536`
- **Output Projection** (`W_o`): `512 × 512 = 262,144`

Attention Total = 262,144 + 65,536 + 65,536 + 262,144 = 655,360

#### b. LayerNorms (x2)

Each LayerNorm: `2 × d_model = 1024`

---

#### 🧠 3. MoE FFN (per layer)

Each expert:

- `W1`: `512 × 2048 = 1,048,576`
- `W2`: `2048 × 512 = 1,048,576`
- Biases: `2048 + 512 = 2560`

Per Expert Total = 2,099,712
Total for 8 Experts = 8 × 2,099,712 = 16,797,696

---

#### 🧠 4. Total Parameters per Transformer Layer

Layer Total = Attention + MoE + LayerNorms
= 655,360 + 16,797,696 + 1,024
= 17,454,080

---

#### 🧠 5. Total for All Transformer Layers

Total = 8 × 17,454,080 = 139,632,640

---

#### 🧠 6. Final Output Layer

Output Head = d_model × vocab_size
= 512 × 32000
= 16,384,000

---

#### ✅ Final Total Parameter Count

| Component              | Count       |
| ---------------------- | ----------- |
| Token Embedding        | 16,384,000  |
| Transformer Layers     | 139,632,640 |
| Output Projection Head | 16,384,000  |

---

Total = 16,384,000 + 139,632,640 + 16,384,000
= `172,400,640 parameters`

#### 🧾 Summary

- **Total Trainable Parameters**: **172.4M**
- **Optimizer**: AdamW (adds states, not parameters)
- **KV Cache & Sliding Window**: Runtime memory only
