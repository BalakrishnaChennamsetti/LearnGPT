import tiktoken, torch
tokenizer = tiktoken.encoding_for_model("gpt-4")
print(tokenizer.decode([13347]))