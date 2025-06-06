import re
from asyncio.log import logger

class DataCleanAndPrepare:

    def __init__(self, raw_data: str):
        self.raw_data = raw_data 

    def clean(self)-> str:
        try:
            if(self.raw_data == None):
                raise(DataCleanAndPrepare("The is not available or invalid data"))
            print(re.sub(r"[^a-zA-Z0-9,. ]", "", self.raw_data))
            return re.sub(r"[^a-zA-Z0-9,. ]", "", self.raw_data)
        except DataCleanAndPrepare:
            logger.info(DataCleanAndPrepare)

    
# print(DataCleanAndPrepare("This ;;,/is the ").clean())