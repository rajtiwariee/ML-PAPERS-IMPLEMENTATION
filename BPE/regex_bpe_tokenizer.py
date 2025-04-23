
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import json
from tqdm import tqdm
import regex as re #more better than re

class RegexBPETokenizer:

  def __init__(self, pattern = None):
    self.pattern = pattern if pattern is not None else GPT4_SPLIT_PATTERN
    self.compiled_pattern = re.compile(self.pattern)
    self.merges = {}
    self.vocab = {}
    self.special_tokens = {}
    self.inverse_special_tokens = {}

  def get_stats(self, tokens, stats:dict):
    """
    Calculates the frequency of each pair of consecutive tokens in the input list.

    Args:
        tokens: A list of tokens (e.g., integers representing bytes).

    Returns:
        A dictionary where keys are tuples representing token pairs and values are their counts.
    """
    for p in zip(tokens, tokens[1:]):
      stats[p] = stats.get(p, 0) + 1 # (pair, count)
    return stats

  def merge(self, tokens, pair, idx):
    """
    Merges occurrences of a specific token pair in a list of tokens with a new token ID.

    Args:
        tokens: A list of tokens.
        pair: A tuple representing the token pair to merge.
        idx: The new token ID to replace the merged pair.

    Returns:
        A new list of tokens with the specified pair merged.
    """
    new_ids = []
    i = 0
    while i < len(tokens):

      if i < len(tokens) - 1 and pair[0] == tokens[i] and pair[1] == tokens[i+1]:
        new_ids.append(idx)
        i+=2
      else:
        new_ids.append(tokens[i])
        i+=1

    return new_ids

  def register_special_tokens(self, special_tokens:dict):
      """special_tokens is a dictionary of str -> int
      example: {"<|endoftext|>": 100257}"""
      self.special_tokens = special_tokens
      self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

  def _train(self, text, vocab_size, verbose = False):
    """
    Trains the BPE tokenizer on a given text, learning merges up to a specified vocabulary size.

    Args:
        text: The text to train on.
        vocab_size: The desired vocabulary size.
        verbose:  Not used.  Consider removing if not needed.
    """
    assert vocab_size >= 256
    vocab_size = vocab_size - 256

    text= re.findall(self.compiled_pattern,text)
    #1.encode the text convert in bytes with utf-8
    ids = [list(ch.encode('utf-8')) for ch in text]

    merges = {} #(int,int) -> int
    vocab = {idx : bytes([idx]) for idx in range(256)}
    for i in tqdm(range(vocab_size)):
      stats = {}
      for chunk_ids in ids:
        self.get_stats(chunk_ids, stats)

      if not stats:
        break
      # find the pair with the highest count
      pair = max(stats, key=stats.get)
      # mint a new token: assign it the next available id
      idx = 256 + i
      # replace all occurrences of pair in ids with idx
      ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
      # save the merge
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
      # prints
      if verbose:
          print(f"merge {i+1}/{vocab_size}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

    #save to class variable
    self.merges = merges
    self.vocab = vocab

  def _encode_chunk(self, text_bytes):
      # return the token ids
      # let's begin. first, convert all bytes to integers in range 0..255
      ids = list(text_bytes)

      while len(ids) >= 2:
          stats = {}
          # find the pair with the lowest merge index
          stats = self.get_stats(ids,stats)
          pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
          # subtle: if there are no more merges available, the key will
          # result in an inf for every single pair, and the min will be
          # just the first pair in the list, arbitrarily
          # we can detect this terminating case by a membership check
          if pair not in self.merges:
              break # nothing else can be merged anymore
          # otherwise let's merge the best pair (lowest merge index)
          idx = self.merges[pair]
          ids = self.merge(ids, pair, idx)
      return ids


  def encode_ordinary(self, text):
      """Encoding that ignores any special tokens."""
      # split text into chunks of text by categories defined in regex pattern
      text_chunks = re.findall(self.compiled_pattern, text)
      # all chunks of text are encoded separately, then results are joined
      ids = []
      for chunk in text_chunks:
          chunk_bytes = chunk.encode("utf-8") # raw bytes
          chunk_ids = self._encode_chunk(chunk_bytes)
          ids.extend(chunk_ids)
      return ids

  def encode(self, text,allowed_special = "none_raise"):
    """Unlike encode_ordinary this handles special tokens
     allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
      if none_raise, then an error is raised if any special token is encountered in text
      this is the default tiktoken behavior right now as well
      any other behavior is either annoying, or a major footgun
    """
    special = None
    if allowed_special == "all":
      special = self.special_tokens
    elif allowed_special == "none":
        special = {}
    elif allowed_special == "none_raise":
        special = {}
        assert all(token not in text for token in self.special_tokens)
    elif isinstance(allowed_special, set):
        special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
    else:
        raise ValueError(f"allowed_special={allowed_special} not understood")
    if not special:
        # shortcut: if no special tokens, just use the ordinary encoding
        return self.encode_ordinary(text)
    # otherwise, we have to be careful with potential special tokens in text
    # we handle special tokens by splitting the text
    # based on the occurrence of any exact match with any of the special tokens
    # we can use re.split for this. note that surrounding the pattern with ()
    # makes it into a capturing group, so the special tokens will be included
    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
    special_chunks = re.split(special_pattern, text)
    # now all the special characters are separated from the rest of the text
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for part in special_chunks:
        if part in special:
            # this is a special token, encode it separately as a special case
            ids.append(special[part])
        else:
            # this is an ordinary sequence, encode it normally
            ids.extend(self.encode_ordinary(part))
    return ids

  def decode(self, tokens:list[int]) -> str:
    """
    Decodes a list of token IDs back into a text string.

    Args:
        tokens: A list of integer token IDs.

    Returns:
        The decoded text string.

    Raises:
        KeyError: If a token ID is not found in the vocabulary.
    """
    try:
      part_bytes = []
      for idx in tokens:
        if idx in self.vocab:
          part_bytes.append(self.vocab[idx])
        elif idx in self.inverse_special_tokens:
          part_bytes.append(self.inverse_special_tokens[idx].encode('utf‑8'))
        else:
          print(f'Invalid token id')

      tokens_bytes = b"".join(idx for idx in part_bytes)
      text = tokens_bytes.decode('utf-8')
      return text
    except KeyError as e:
      raise KeyError(f"Token ID {e} not found in vocabulary.  Your tokenizer might not be trained or you may be trying to decode tokens that were not created by this instance")

  def save(self, filename):
    """
    Saves the tokenizer's vocabulary and merges to a JSON file.

    Args:
        filename: The name of the file to save the tokenizer data to.
    """
    # We need to serialize bytes in vocab to lists of integers
    vocab_serializable = {k: list(v) for k, v in self.vocab.items()}

    data = {
        "merges": list(self.merges.items()),  # Convert dict to list of (key, value) pairs for serialization
        "vocab": vocab_serializable
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Tokenizer saved to {filename}")


  def load(self, filename):
    """
    Loads the tokenizer's vocabulary and merges from a JSON file.

    Args:
        filename: The name of the file to load the tokenizer data from.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert keys in merges back to tuples
    self.merges = {tuple(map(int, key)): value for key, value in data["merges"]}

    # Deserialize lists of integers back to bytes in vocab
    self.vocab = {k: bytes(v) for k, v in data["vocab"].items()}

    print(f"Tokenizer loaded from {filename}")


#Example usage and test
if __name__ == '__main__':
  tokenizer = RegexBPETokenizer()
  vocab_size = 10_000 # Adjust as needed
  tokenizer._train(taylor_text, vocab_size)
