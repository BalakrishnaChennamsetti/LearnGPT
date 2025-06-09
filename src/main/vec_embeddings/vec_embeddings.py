import torch
import torch.nn as nn

from configuration.model_config import ModelConfig

class VecEmbeddings:
    
    def __init__(self, batch_x, batch_y, block_size = 16):
        modelconfig = ModelConfig()
        self.batch_x = batch_x
        self.batch_y = batch_y
        self.block_size = block_size
        self.vocab_size = modelconfig.vocab_size
        self.emb_dim = modelconfig.emb_dim
    
    def embed(self):
        vocab_size = max(self.batch_x.max() , self.batch_y.max(), self.vocab_size) + 1
        embedding_dim = self.emb_dim

        embedding = nn.Embedding(vocab_size, embedding_dim)
        X_vector = embedding(self.batch_x)
        y_vector = embedding(self.batch_y)

        return X_vector, y_vector