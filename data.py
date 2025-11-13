import os
import torch
import numpy as np

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
    def encode(self, text):
        b = text.encode('utf-8')
        return np.frombuffer(b, dtype=np.uint8).astype(np.int64)
    def decode(self, ids):
        a = np.array(ids, dtype=np.uint8).tobytes()
        return a.decode('utf-8', errors='ignore')

class LMTextDataset:
    def __init__(self, path, split_ratios=(0.85, 0.10, 0.05), lowercase=False):
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        if lowercase:
            txt = txt.lower()
        self.tokenizer = ByteTokenizer()
        ids = self.tokenizer.encode(txt)
        self.tokens = torch.from_numpy(ids)
        n = self.tokens.size(0)
        a = int(n * split_ratios[0])
        b = int(n * (split_ratios[0] + split_ratios[1]))
        self.train_tokens = self.tokens[:a]
        self.val_tokens = self.tokens[a:b]
        self.test_tokens = self.tokens[b:]
    def _stream(self, split):
        if split == 'train':
            return self.train_tokens
        if split == 'val':
            return self.val_tokens
        return self.test_tokens
    def get_batch(self, split, batch_size, block_size, device):
        s = self._stream(split)
        L = s.size(0) - block_size - 1
        idx = torch.randint(0, L, (batch_size,), device=device)
        x = torch.stack([s[i:i+block_size] for i in idx]).to(device)
        y = torch.stack([s[i+1:i+block_size+1] for i in idx]).to(device)
        return x.long(), y.long()