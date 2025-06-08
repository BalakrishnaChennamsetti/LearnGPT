from dataclasses import dataclass

@dataclass
class ModelConfig:
    batch_size: int = 16
    max_len: int = 128
    stride: int  = 64
    vocab_size: int = 50257   # Vocabulary size
    context_length: int = 128 # Shortened context length (orig: 1024)
    emb_dim: int = 768        # Embedding dimension
    n_heads: int  = 12        # Number of attention heads
    n_layers: int = 12        # Number of layers
    drop_rate: int = 0.1 
    shuffle: bool = True     # Dropout rate
    qkv_bias: bool = False 
    drop_last: bool = True 
    num_workers: int = 0
