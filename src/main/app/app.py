import sys
import os

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion.raw_text_extract_from_git_repos import RawDataIngestion
from data_ingestion.data_cleaning_and_preparation import DataCleanAndPrepare
from tokenization.bpe_tokenizer import Tokenizer
from dataset_preparation.prepare_data import PrepareData
from dataset_preparation.dataset import Dataset
from configuration.model_config import ModelConfig

class App:

    def __init__(self):

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        modelconfig = ModelConfig()
        self.dataset = Dataset()

        #Config Parameters
        self.batch_size = modelconfig.batch_size
        self.max_length = modelconfig.max_len
        self.stride = modelconfig.stride
        self.context_length  = modelconfig.context_length
        self.shuffle = modelconfig.shuffle
        self.drop_last = modelconfig.drop_last
        self.num_workers = modelconfig.num_workers

    def run(self):

        # 1. Raw data extarction and data ingestion
        raw_data_ingestion = RawDataIngestion()
        raw_data_list = raw_data_ingestion.extract_raw_data()

          #2.  Data cleaning and data preparation
        data_clean_and_prepare = DataCleanAndPrepare(raw_data_list)
        clean_data = data_clean_and_prepare.clean()

        #3. Tokenization
        tokenizer = tiktoken.get_encoding("cl100k_base")
        # print(clean_data)
        total_characters = len(clean_data)
        total_tokens = len(tokenizer.encode(clean_data))
        print("Tokend Total:", total_tokens,"\n\n\n")
        print("Characters:", total_characters)
        # print("Tokens:", total_tokens)

        # dataset = PrepareData(clean_data, self.tokenizer, max_length, stride)

        # Train/validation ratio
        train_ratio = 0.90
        split_idx = int(train_ratio * total_characters)
        train_data = clean_data[:split_idx]
        val_data = clean_data[split_idx:]
        print(len(train_data), len(val_data))
        print(len(train_data)+len(val_data)-total_characters)


        torch.manual_seed(123)
        train_loader = self.dataset.initiate_analysis(
                        train_data,
                        tokenizer = tokenizer,
                        batch_size=self.batch_size,
                        max_length=self.max_length,
                        stride=self.context_length,
                        drop_last=True,
                        shuffle=True,
                        num_workers=0
                        )   

        val_loader =  self.dataset.initiate_analysis(
                        val_data,
                        tokenizer = tokenizer,
                        batch_size=self.batch_size,
                        max_length=self.max_length,
                        stride=self.context_length,
                        drop_last=False,
                        shuffle=False,  
                        num_workers=0
                        )
        
        # Sanity check

        assert total_tokens * (train_ratio) > self.context_length, "Not enough tokens for the training loader. Try to lower the `GPT_CONFIG_124M['context_length']` or increasethe `training_ratio`"
        assert total_tokens * (1-train_ratio) > self.context_length, "Not enough tokens for the validation loader. Try to lower the `GPT_CONFIG_124M['context_length']` or decrease the `training_ratio`"

        print("Train loader:")
        for x, y in train_loader:
            print(x.shape, y.shape)

        print("\nValidation loader:")
        for x, y in val_loader:
            print(x.shape, y.shape)


        print(len(train_loader))
        print(len(val_loader))

        train_tokens = 0
        for input_batch, target_batch in train_loader:
            train_tokens += input_batch.numel()

        val_tokens = 0
        for input_batch, target_batch in val_loader:
            val_tokens += input_batch.numel()

        print("Training tokens:", train_tokens)
        print("Validation tokens:", val_tokens)
        print("All tokens:", train_tokens + val_tokens)

        # print(train_loader.dataset.__getitem__.__get__)
        # print(val_loader.dataset.__getitem__)

        # batch_x, batch_y = next(iter(train_loader))
        # print("X:", batch_x)
        # print("Y:", batch_y)

    
app = App()
app.run()