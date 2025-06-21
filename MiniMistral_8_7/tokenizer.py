from sentencepiece import SentencePieceProcessor
from typing import List, Optional, Tuple, Union
from pathlib import Path


class Tokenizer:
    def __init__(self, model_path: Union[str,Path]):
        self.model_path = model_path
        self._model = SentencePieceProcessor(model_file=str(model_path))
        assert self._model.vocab_size == self._model.get_piece_size()
        
    @property
    def vocab_size(self) -> int:
        return self._model.vocab_size()
    
    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)