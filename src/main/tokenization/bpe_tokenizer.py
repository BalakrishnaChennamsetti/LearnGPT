import tiktoken

class Tokenizer():

    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def tokenize(self, data: str) -> list[int]:
        self.tokenized_data = self.tokenizer.encode(data)
        return self.tokenized_data

