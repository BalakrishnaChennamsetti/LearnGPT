import tiktoken

from configuration.model_config import ModelConfig

class Tokenizer:

    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.model_config = ModelConfig()
    
    def tokenize(self, data: str) -> list[int]:
        self.tokenized_data = self.tokenizer.encode(data)
        return self.tokenized_data

    def training_data(self, tokenized_data: list[int]):
        k = self.model_config.window_size
        n = len(tokenized_data)
        print(n, k)
        training_data_list_X = []
        training_data_list_y = []
        for i in range(0, n-k, k):
            # if(i==0):
                # print(tokenized_data[i:k])
            training_data_list_X.append(tokenized_data[i:i+k])
            training_data_list_y.append([tokenized_data[i+k+1]])
        return training_data_list_X, training_data_list_y
