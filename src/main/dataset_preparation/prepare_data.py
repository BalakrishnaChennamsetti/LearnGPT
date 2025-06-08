import torch
from torch.utils.data import Dataset, DataLoader

from configuration.model_config import ModelConfig
from utils.special_tokens import SpecialTokenIDs

class PrepareData(Dataset):
     
    def __init__(self, txt, tokenizer, max_length, stride):
        self.training_data_list_X = []
        self.training_data_list_y = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.training_data_list_X.append(torch.tensor(input_chunk))
            self.training_data_list_y.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.training_data_list_X)

    def __getitem__(self, idx):
        return self.training_data_list_X[idx], self.training_data_list_y[idx]