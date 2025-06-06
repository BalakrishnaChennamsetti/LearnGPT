from dataclasses import dataclass

@dataclass
class ModelConfig:
    contex_size: int = 128
    embedding_size: int = 1024
    window_size: int = 8
