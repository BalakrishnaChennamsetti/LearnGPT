import tiktoken
import torch

class Tokenizer():

    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt2")
    
    def tokenize(self, data: str):
        encoded = self.tokenizer.encode(data, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor
    def to_text(self, idx):
        flat = idx.squeeze(0) # remove batch dimension
        return self.tokenizer.decode(flat.tolist())



