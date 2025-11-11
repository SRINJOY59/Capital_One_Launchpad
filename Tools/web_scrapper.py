import os
import time
from typing import List, Dict, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

class WebScrapper:
    def __init__(self, headless: bool = True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options=chrome_options)

    def fetch_page(self, url: str, wait_time: int = 3) -> str:
        self.driver.get(url)
        time.sleep(wait_time)
        return self.driver.page_source

    def extract_table(self, url: str, table_selector: str = "table") -> List[List[str]]:
        html = self.fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one(table_selector)
        rows = []
        if table:
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
        return rows

    def extract_text(self, url: str, selector: str) -> List[str]:
        html = self.fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements]

    def extract_links(self, url: str, selector: str = "a") -> List[str]:
        html = self.fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)
        return [el.get("href") for el in elements if el.get("href")]

    def close(self):
        self.driver.quit()

def scrape_agri_prices(url: str, table_selector: str = "table") -> List[Dict[str, Any]]:
    scrapper = WebScrapper()
    rows = scrapper.extract_table(url, table_selector)
    headers = rows[0] if rows else []
    data = []
    for row in rows[1:]:
        item = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
        data.append(item)
    scrapper.close()
    return data

def scrape_policy_updates(url: str, selector: str = ".policy-update") -> List[str]:
    scrapper = WebScrapper()
    updates = scrapper.extract_text(url, selector)
    scrapper.close()
    return updates

def scrape_links(url: str, selector: str = "a") -> List[str]:
    scrapper = WebScrapper()
    links = scrapper.extract_links(url, selector)
    scrapper.close()
    return links
