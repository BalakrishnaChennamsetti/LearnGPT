import torch
from torch.utils.data import Dataset, DataLoader

from data_ingestion.raw_text_extract_from_git_repos import RawDataIngestion
from data_ingestion.data_cleaning_and_preparation import DataCleanAndPrepare
from tokenization.bpe_tokenizer import Tokenizer
from dataset_preparation.prepare_data import PrepareData
from configuration.model_config import ModelConfig

class Dataset:
    def __init__(self):
                pass

    def initiate_analysis(sel, data, tokenizer, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

        dataset = PrepareData(
               data,
               tokenizer=tokenizer,
               max_length= max_length,
               stride = stride
        )
        
        dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
        )

        return dataloader
