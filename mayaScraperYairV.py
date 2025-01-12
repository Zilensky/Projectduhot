import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Initialize WebDriver (for Chrome)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))



def download_pdf(pdf_link, download_directory="downloads"):
    """
    Download the PDF from a given link and save it to the specified directory.

    :param pdf_link: The direct URL to the PDF.
    :param download_directory: Directory where the PDF will be saved.
    """
    # Ensure the download directory exists
    os.makedirs(download_directory, exist_ok=True)

    # Full URL for the PDF
    full_url = f"https://maya.tase.co.il/reports/{pdf_link}?view=finance"

    # Navigate to the PDF URL
    driver.get(full_url)

    # Wait for the page to load (adjust the time as needed)
    time.sleep(5)

    # Example: Get the current URL after redirection (if necessary)
    current_url = driver.current_url

    # Suggesting filename for saving
    filename = f"{pdf_link}.pdf"
    filepath = os.path.join(download_directory, filename)

    # Using requests (instead of Selenium) for downloading the file
    import requests
    response = requests.get(current_url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filepath}")
    else:
        print(f"Failed to download {pdf_link}: {response.status_code}")


# Iterate over the PDF links and download each
'''for link in pdf_links:
    download_pdf(link)'''

download_pdf("1829")

# Close the browser after downloading
driver.quit()
