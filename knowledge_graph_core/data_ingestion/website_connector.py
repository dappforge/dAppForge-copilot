from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from chromedriver_autoinstaller import install as chromedriver_autoinstaller_install
from bs4 import BeautifulSoup


def scrape_website(url: str, max_depth: int = 10):
    # Set up Chrome options
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--headless")  # Run Chrome in headless mode for scraping without a GUI
    
    # Ensure the ChromeDriver is installed
    chromedriver_autoinstaller_install()
    
    # Set up the Chrome WebDriver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    
    try:
        # Initialize the scraper with the given URL and maximum depth
        scraper = WholeSiteReader(
            prefix=url,
            max_depth=max_depth,  # Set the maximum depth for scraping
            driver=driver  # Pass the configured WebDriver
        )
        
        # Start scraping from the base URL
        all_docs = scraper.load_data(base_url=url)  # Load data from the base URL
        
    except Exception as e:
        print(f"An error occurred while scraping the website: {e}")
        all_docs = None
    
    finally:
        # Quit the WebDriver to free resources
        driver.quit()
    
    return all_docs
