import torch


class CharTokenizer:
    def __init__(self, init_text: str):
        self.chars = sorted(list(set(init_text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

    def encode(self, chars: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx[c] for c in chars])

    def decode(self, idxs: torch.Tensor) -> str:
        return "".join([self.idx_to_char[i.item()] for i in idxs])
