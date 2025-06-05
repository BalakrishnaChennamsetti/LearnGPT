from data_ingestion.raw_text_extract_from_git_repos import RawDataIngestion
from data_ingestion.data_cleaning_and_preparation import DataCleanAndPrepare

#1. Raw data extarction and Data Ingestion
raw_data_ingestion = RawDataIngestion()
raw_data_list = raw_data_ingestion.extract_raw_data()

#2.  Data cleaning and data preparation
data_clean_and_prepare = DataCleanAndPrepare(raw_data_list)
clean_data = data_clean_and_prepare.clean()

#3. Embeddings creation
#4. Tokenization

#5. Transformer Block

#6. Ouput block

#7. Validation

#8.Inferancing