import json
from tqdm import tqdm

class BasicBPETokenizer:

  def __init__(self):
    self.merges = {}
    self.vocab = {}

  def get_stats(self, tokens):
    """
    Calculates the frequency of each pair of consecutive tokens in the input list.

    Args:
        tokens: A list of tokens (e.g., integers representing bytes).

    Returns:
        A dictionary where keys are tuples representing token pairs and values are their counts.
    """
    stats = {}
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

  def _train(self, text, vocab_size, verbose = False):
    """
    Trains the BPE tokenizer on a given text, learning merges up to a specified vocabulary size.

    Args:
        text: The text to train on.
        vocab_size: The desired vocabulary size.
        verbose:  Not used.  Consider removing if not needed.
    """
    assert vocab_size >= 256
    #1.encode the text convert in bytes with utf-8
    tokens = text.encode('utf-8')
    tokens = list(map(int, tokens))
    #2.1st will construct the merge
    vocab_size = vocab_size - 256

    for idx in tqdm(range(vocab_size), desc="Training BPE"):
      stats = self.get_stats(tokens)
      pair = max(stats, key=stats.get)
      idx = 256 + idx
      tokens = self.merge(tokens, pair, idx)
      self.merges[pair] = idx

    #2. use these merges to create the vocab
    self.vocab = {idx : bytes([idx]) for idx in range(256)}
    for (p0,p1), idx in tqdm(self.merges.items(), desc="Creating Vocabulary"):
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    return None

  def encode(self, text: str) -> list[int]:
    """
    Encodes a text string into a list of token IDs using the learned BPE merges.

    Args:
        text: The text to encode.

    Returns:
        A list of integer token IDs.
    """
    tokens = list(text.encode('utf-8'))

    while len(tokens) >= 2:
      stats = self.get_stats(tokens)
      #Crucial: Use merges.get(p, float('inf')) for unknown pairs
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      if pair not in self.merges:
        break # nothing else can be merged
      idx = self.merges[pair]
      tokens = self.merge(tokens, pair, idx)
    return tokens

  def decode(self, tokens: list[int]) -> str:
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
      tokens_bytes = b"".join(self.vocab[idx] for idx in tokens)
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
  tokenizer = BasicBPETokenizer()
  text = "This is a test sentence. This is another test sentence.  The cat sat on the mat."
  vocab_size = 300 # Adjust as needed
  tokenizer._train(text, vocab_size)

  # Save the tokenizer
  tokenizer.save("tokenizer.json")

  # Load the tokenizer
  loaded_tokenizer = BasicBPETokenizer()
  loaded_tokenizer.load("tokenizer.json")

  # Test the loaded tokenizer
  unseen_text = "The dog also sat on the mat."
  encoded_unseen = loaded_tokenizer.encode(unseen_text)
  print(f"Encoded unseen text (loaded tokenizer): {encoded_unseen}")
  decoded_unseen = loaded_tokenizer.decode(encoded_unseen)
  print(f"Decoded unseen text (loaded tokenizer): {decoded_unseen}")
