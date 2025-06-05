import re

class DataCleanAndPrepare:

    def __init__(self, raw_data: str):
        self.raw_data = raw_data 

    def clean(self)-> str:
        print(re.sub(r"[^a-zA-Z0-9,. ]", "", self.raw_data))
        return re.sub(r"[^a-zA-Z0-9,. ]", "", self.raw_data)
    
# print(DataCleanAndPrepare("This ;;,/is the ").clean())