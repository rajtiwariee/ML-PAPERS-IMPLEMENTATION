# #!/usr/bin/env python
# # tokenize_tinystories_and_dataloader.py
# # ---------------------------------------------------------------
# # 0.  Imports & constants
# # ---------------------------------------------------------------
# import os, multiprocessing as mp, random, numpy as np, tiktoken
# from pathlib import Path
# from datasets import load_dataset
# from tqdm import tqdm
# import torch
# from torch.utils.data import IterableDataset, DataLoader

# OUT_DIR    = Path("tinystories_npy")
# SHARD_SIZE = 100_000_000           # 100 M tokens → ~200 MB per file
# TOK        = tiktoken.get_encoding("gpt2")
# EOT        = TOK._special_tokens["<|endoftext|>"]
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # ---------------------------------------------------------------
# # 1.  Load TinyStories (streaming)
# # ---------------------------------------------------------------
# train_stream = load_dataset("roneneldan/TinyStories",
#                             split="train",      streaming=True)
# val_stream   = load_dataset("roneneldan/TinyStories",
#                             split="validation", streaming=True)

# def concat_streams():                # ensures val shards first
#     for doc in val_stream:
#         yield doc, "val"
#     for doc in train_stream:
#         yield doc, "train"

# # ---------------------------------------------------------------
# # 2.  Encode helper (multiprocess-safe)
# # ---------------------------------------------------------------
# def encode(pair):
#     doc, split = pair
#     ids = TOK.encode_ordinary(doc["text"]) + [EOT]
#     return np.asarray(ids, dtype=np.uint16), split

# # ---------------------------------------------------------------
# # 3.  Shard writer
# # ---------------------------------------------------------------
# def build_shards():
#     buf, filled, shard_idx = np.empty(SHARD_SIZE, np.uint16), 0, 0
#     curr_split = "val"
#     pbar = tqdm(unit="tok", desc="Tokenising TinyStories")

#     with mp.Pool(max(1, os.cpu_count() // 2)) as pool:
#         for ids, split in pool.imap(encode, concat_streams(), chunksize=32):
#             if split != curr_split:
#                 if filled:
#                     np.save(OUT_DIR / f"tinystories_{curr_split}_{shard_idx:04d}.npy", buf[:filled])
#                     print(f"📝 wrote shard #{shard_idx} ({filled:,} tok) → {curr_split}")
#                     shard_idx += 1
#                     filled = 0
#                 curr_split = split
#             idx = 0
#             while idx < len(ids):
#                 take = min(SHARD_SIZE - filled, len(ids) - idx)
#                 buf[filled:filled+take] = ids[idx:idx+take]
#                 filled += take
#                 idx    += take
#                 pbar.update(take)
#                 if filled == SHARD_SIZE:
#                     np.save(OUT_DIR / f"tinystories_{curr_split}_{shard_idx:04d}.npy", buf)
#                     print(f"📝 wrote shard #{shard_idx} ({filled:,} tok) → {curr_split}")
#                     shard_idx += 1
#                     filled = 0
#     if filled:
#         np.save(OUT_DIR / f"tinystories_{curr_split}_{shard_idx:04d}.npy", buf[:filled])
#         print(f"📝 wrote shard #{shard_idx} ({filled:,} tok) → {curr_split}")

# # ---------------------------------------------------------------
# # 4.  IterableDataset (Corrected Version)
# # ---------------------------------------------------------------
# class NpyShardDataset(IterableDataset):
#     def __init__(self, root_dir: str, split: str, seq_len: int = 2048):
#         self.files = sorted(Path(root_dir).glob(f"tinystories_{split}_*.npy"))
#         if not self.files:
#             raise RuntimeError(f"No shards for split='{split}' in {root_dir}")
#         self.seq_len = seq_len
#         self.split = split  # Storing split type is good practice!

#     def __iter__(self):
#         worker = torch.utils.data.get_worker_info()
#         files = self.files if worker is None else self.files[worker.id::worker.num_workers]

#         # 1. Shuffle shards for the training set to vary the order of data chunks.
#         if self.split == "train":
#             random.shuffle(files)

#         buf = np.empty(0, dtype=np.uint16)

#         for f in files:
#             arr = np.load(f, mmap_mode="r")
            
#             # 2. For training, start at a random offset to create different sequences each epoch.
#             #    For validation, always start at 0 for a deterministic, repeatable evaluation.
#             if self.split == "train":
#                 idx = random.randint(0, self.seq_len)
#             else:
#                 idx = 0

#             while idx < len(arr):
#                 need = self.seq_len + 1 - len(buf)
#                 take = min(need, len(arr) - idx)
#                 buf = np.concatenate((buf, arr[idx:idx+take])); idx += take

#                 if len(buf) >= self.seq_len + 1:
#                     chunk, buf = buf[:self.seq_len+1], buf[self.seq_len+1:]
#                     # .copy() is important to ensure x and y are independent tensors
#                     x = torch.from_numpy(chunk[:-1].copy()).long()
#                     y = torch.from_numpy(chunk[1:].copy()).long()
#                     yield x, y

# # ---------------------------------------------------------------
# # 5.  Build shards & quick test
# # ---------------------------------------------------------------
# if __name__ == "__main__":
#     print("→ Building TinyStories shards …")
#     build_shards()

#     SEQ_LEN = 2048
#     BATCH   = 4
#     train_loader = DataLoader(
#         NpyShardDataset("tinystories_npy", "train", seq_len=SEQ_LEN),
#         batch_size=BATCH, num_workers=0)
#     val_loader   = DataLoader(
#         NpyShardDataset("tinystories_npy", "val",   seq_len=SEQ_LEN),
#         batch_size=BATCH, num_workers=0, drop_last=True)

#     xb, yb = next(iter(train_loader))
#     print("Train batch:", xb.shape, yb.shape)





from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


dataset=load_dataset("roneneldan/TinyStories",split="train[:50000]")
dataset
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")  # or "meta-llama/Llama-2-7b"
tokens = tokenizer("The cat sat on the mat.", return_tensors="pt")
train_data=dataset[:50000]
val_data=dataset[50000:51000]

vocab_size=tokenizer.vocab_size
class Tiny_dataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=150):
        """
        Initializes a TinyDataset instance.

        Args:
            data (dict): A dictionary containing the dataset with a "text" key.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the text data.
            max_seq_length (int, optional): The maximum sequence length for tokenization. Defaults to 150.

        Attributes:
            data (dict): The dataset with text data.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding.
            encoded_texts (list): A list of encoded text sequences.
            max_seq_length (int): The maximum sequence length for tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Encode text data and store as encoded texts
        self.encoded_texts = [
            tokenizer.encode(
                text,
                truncation=True,
                max_length=max_seq_length + 1,  # Add 1 to handle the shifting of labels
                padding=False
            ) for text in tqdm(data["text"], "Encoding")
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        
        # Create input_ids and labels
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)  # Exclude last token for input
        labels = torch.tensor(encoded[1:], dtype=torch.long)  # Exclude first token for labels

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def __len__(self):
        return len(self.encoded_texts)
  


train_dataset=Tiny_dataset(data=train_data,tokenizer=tokenizer)
val_dataset=Tiny_dataset(data=val_data,tokenizer=tokenizer)



from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    """
    # Extract 'input_ids' and 'labels' tensors from the batch
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad the input_ids and labels separately
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return padded_input_ids, padded_labels





num_workers=2
batch_size=8

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1, #for inference
    num_workers=num_workers,
    drop_last=False,
    collate_fn=collate_fn
)
print(len(val_loader))
