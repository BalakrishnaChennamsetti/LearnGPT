import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion.raw_text_extract_from_git_repos import RawDataIngestion
from data_ingestion.data_cleaning_and_preparation import DataCleanAndPrepare
from tokenization.bpe_tokenizer import Tokenizer


#1. Raw data extarction and Data Ingestion
raw_data_ingestion = RawDataIngestion()
raw_data_list = raw_data_ingestion.extract_raw_data()

#2.  Data cleaning and data preparation
data_clean_and_prepare = DataCleanAndPrepare(raw_data_list)
clean_data = data_clean_and_prepare.clean()

#3. Tokenization
tokenizer = Tokenizer()
tokenized_data = tokenizer.tokenize(clean_data)
# print(tokenized_data)
training_data_list_X, training_data_list_y = tokenizer.training_data(tokenized_data)
print(training_data_list_y)
#4. Embeddings creation
#5. Transformer Block

#6. Ouput block

#7. Validation

#8.Inferancing