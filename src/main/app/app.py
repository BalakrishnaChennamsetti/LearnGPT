from asyncio.log import logger
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
from vec_embeddings.vec_embeddings import VecEmbeddings

from transformerblock.gpt_architecture import GPTModel
from transformerblock.load_weights import LoadWeights

class App:

    def __init__(self):

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        model_config = ModelConfig()
        self.dataset = Dataset()
        self.gpt_model = GPTModel(model_config)
        self.gpt_model.eval()
        self.load_weights = LoadWeights()
        self.device = "cpu"

        #Config Parameters
        self.batch_size = model_config.batch_size
        self.max_length = model_config.max_len
        self.stride = model_config.stride
        self.context_length  = model_config.context_length
        self.shuffle = model_config.shuffle
        self.drop_last = model_config.drop_last
        self.num_workers = model_config.num_workers

    def load_model_with_pretrain_weights(self, path):
        # logits = self.gPTModel.forward(in_idx)
        settings, params = self.load_weights.load_gpt2(path)
        self.load_weights.load_weights_into_gpt2(self.gpt_model, params)
        logger.log(1, "GPT2 Weights Successfully Loaded into Local LearnGPT Model...", "", exc_info=1)
        self.gpt_model.to(self.device);
        # return logits
        
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
                        drop_last=True,
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
        
        batch_x, batch_y = next(iter(train_loader))
        print("X:", batch_x)
        print("Y:", batch_y)
        
        ## Training embedding vectors creation
        all_train_X_vectors = []
        all_train_y_vectors = []

        for batch_x, batch_y in train_loader:
            self.vecembeddings = VecEmbeddings(batch_x, batch_y)
            X_vector, y_vector = self.vecembeddings.embed()
            all_train_X_vectors.append(X_vector)
            all_train_y_vectors.append(y_vector)
            # Optional: check values
            print("X_vector shape:", X_vector.shape)
            print("y_vector shape:", y_vector.shape)


        ## Testing vector embeddings creation
        all_test_X_vectors = []
        all_test_y_vectors = []

        for batch_x, batch_y in val_loader:
            self.vecembeddings = VecEmbeddings(batch_x, batch_y)
            X_vector, y_vector = self.vecembeddings.embed()
            all_test_X_vectors.append(X_vector)
            all_test_y_vectors.append(y_vector)
            # Optional: check values
            print("X_vector shape:", X_vector.shape)
            print("y_vector shape:", y_vector.shape)
            
        logits = self.load_model_with_pretrain_weights("src/main/resources/gpt2/124M")


    
app = App()
app.run()