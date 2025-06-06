from asyncio.log import logger
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

class RawDataIngestion:
    def __init__(self):
        self.driver = webdriver.Chrome()
    def extract_raw_data(self):
        try:
            self.driver.get("https://github.com/kuemit/txt_book/blob/master/examples/alice_in_wonderland.txt")
            # raise(BaseException)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            textarea = soup.find_all("textarea")
            # textarea = soup.find_all('textarea#read-only-cursor-text-area.react-blob-textarea.react-blob-print-hide')
            final_text = textarea.__str__().split("THE END")[0].splitlines()[1:]
            # logger.info(final_text)
            str_data=""
            for data in final_text:
                str_data+=data
            # print(str_data)
            # logger.info(str_data)
        except BaseException:
            logger.warning("Raw data extraction is failed with ", exc_info=True)
        finally:
            self.driver.quit()
        return str_data

# RawDataIngestion().extract_raw_data()