# def __init__(self):
#         self.tokenizer = Tokenizer()
#         # 1. Raw data extarction and Data Ingestion
#         raw_data_ingestion = RawDataIngestion()
#         raw_data_list = raw_data_ingestion.extract_raw_data()

#         #2.  Data cleaning and data preparation
#         data_clean_and_prepare = DataCleanAndPrepare(raw_data_list)
#         clean_data = data_clean_and_prepare.clean()

#         #3. Tokenization
    
#         tokenized_data = self.tokenize(clean_data)
#         # print(tokenized_data)
#     def create_dataloader_v1(txt, batch_size=4, max_length=256, 
#                          stride=128, shuffle=True, drop_last=True,
#                          num_workers=0):

#         # Initialize the tokenizer
#         tokenizer = tiktoken.get_encoding("gpt2")

#         # # Create dataset
#         # dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
#         dataset = PrepareData(tokenized_data)
#         # Create dataloader
#         dataloader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             drop_last=drop_last,
#             num_workers=num_workers
#         )

#         return dataloader
#     #4. Embeddings creation
#     #5. Transformer Block

#     #6. Ouput block

#     #7. Validation

#     #8.Inferancing