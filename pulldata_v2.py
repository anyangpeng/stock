from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

class Getdata():
    def __init__(self,config):
        self.url = config['url']
        self.downloadPath = config['savingPath']
        
    def _setup(self):
        # changing download folder
        chromeOptions = webdriver.ChromeOptions()
        prefs = {"download.default_directory" : self.downloadPath,
                "download.directory_upgrade": True}
        chromeOptions.add_experimental_option("prefs",prefs)
        self.driver = webdriver.Chrome(service = Service(ChromeDriverManager().install()),
                options = chromeOptions)

    def download(self):
        self._setup()
        self.driver.get(self.url)
        self.driver.set_window_size(1800,1200)
        
        #click on time range
        period = self.driver.find_element("xpath","//div[contains\
                (@class, 'Pos(r) D(ib) C($linkColor) Cur(p)')]")
        time.sleep(1)
        period.click()

        #click on 5year
        time.sleep(1)
        periodButton = self.driver.find_element("xpath","//button[contains\
                (@data-value,'5_Y')]")
        periodButton.click()

        #click on Apply
        applyButton = self.driver.find_element("xpath","//button[starts-with\
                (@class,' Bgc($linkColor) Bdrs(3px)')]")
        applyButton.click()

        #click on download
        saveButton = self.driver.find_element("xpath","//a[contains\
                (@class,'Fl(end) Mt(3px) Cur(p)')]")
        saveButton.click()
        time.sleep(5)#insure complete download

    def close(self):
        print('Download finished.')
        self.driver.close()

if __name__ == '__main__':
    with open('config.json') as configFile:
        config = json.load(configFile)

    data = Getdata(config)
    data.download()
    data.close()
