from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os
import re
import yfinance as yf
import pandas as pd
import openpyxl
# Define the path to your Chrome user profile
chrome_profile_path = os.path.expanduser(r'~\AppData\Local\Google\Chrome\User Data\Default')
chrome_user_data_dir = r"C:\Users\Ascending\AppData\Local\Google\Chrome\User Data"
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--user-data-dir=' + chrome_user_data_dir)


# Initialize the Chrome driver with your profile
driver = webdriver.Chrome(service=Service(), options=chrome_options)

# Navigate to the financial reports page
url = 'https://maya.tase.co.il/reports/finance'
"""driver.get(url)

# Wait for the page to load completely
import time
time.sleep(5)"""

"""# Find all the 'a' tags that contain the links to the reports
report_links = driver.find_elements(By.CSS_SELECTOR, "a.ng-isolate-scope")
pdf_links = []
pattern = re.compile(r'.*/2/\d+$')
# Extract and print the href attributes of each link
for link in report_links:
    href = link.get_attribute('href')
    if href:  # ensure href is not empty
        if "company" in href: # אפשר גם לראות את החברה עצמה אם להוסיף לIF "company" in href
            href = href + "?view=finance"
            print(href)
            pdf_links.append(href)
# Remove duplicates
pdf_links = list(set(pdf_links))
print(pdf_links)
"""
pdf_links = [
    "373", "1829", "1390", "2442", "1461", "444", "1397", "675", "297", "182",
    "1063", "1135", "1682", "1671", "1473", "2134", "2280", "1393", "1001", "1287",
    "2392", "1610", "265", "1451", "1899", "1644", "1425", "1691", "2250", "1696",
    "715", "1307", "1912", "175", "1106", "1715", "1890", "1842", "169", "281",
    "431", "1608", "1566", "1921", "209", "1993", "1862", "1300", "1019", "2357",
    "1833", "282", "341", "1028", "174", "2156", "1480", "1012", "1616", "1036",
    "363", "1152", "1091", "1039", "474", "1040", "1382", "1636", "504", "2252",
    "1978", "2101", "2063", "390", "1729", "2367", "1814", "1553", "1704", "1194",
    "694", "739", "1264", "1804", "501", "368", "387", "749", "1756", "1761",
    "1632", "1328", "589", "654", "2400", "382", "180", "1391", "1863", "1780",
    "720", "1008", "2352", "1683", "1762", "1839", "1581", "2417", "422", "550",
    "313", "1074", "1172", "1979", "1998", "456", "578", "1338", "1824", "755",
    "1891", "1823", "2190", "1622", "383", "770", "2360", "1896", "299", "1975",
    "1884", "1219", "138", "366", "1920", "587", "1122", "1448", "251", "1618",
    "2396", "1465", "2135", "1933", "1925", "1980", "485", "531", "230", "1422",
    "1327", "1832", "2093", "1305", "1855", "1293", "1394", "1859", "593", "1153",
    "530", "1802", "235", "2371", "1361", "1054", "1790", "1827", "424", "1846",
    "1560", "1386", "399", "1269", "1274", "286", "536", "1344", "1940", "259",
    "1452", "454", "759", "771", "506", "1939", "149", "1226", "1310", "1765",
    "126", "1639", "1568", "1731", "448", "271", "2030", "1006", "744", "1130",
    "1918", "1548", "1559", "1741", "1070", "1905", "1510", "1892", "532", "1929",
    "1301", "1820", "1769", "1532", "1960", "1997", "2440", "1604", "1513", "400",
    "1072", "1801", "1312", "2397", "1435", "691", "639", "748", "1867", "1840",
    "2387", "1095", "829", "627", "1858", "1193", "822", "314", "1096", "1778",
    "1294", "1706", "1845", "1551", "1885", "2369", "1826", "1830", "1561", "1187",
    "612", "1459", "767", "1527", "1367", "585", "1708", "161", "1968", "486",
    "1466", "1865", "1772", "350", "416", "371", "2123", "278", "1583", "1515",
    "1907", "1866", "130", "2413", "576", "1722", "823", "351", "1110", "150",
    "1132", "1453", "384", "600", "2028", "629", "462", "1945", "1943", "1081",
    "1936", "1185", "2444", "103", "1796", "1082", "354", "1535", "216", "1982",
    "1870", "666", "1948", "111", "2356", "1903", "1786", "1771", "1825", "1007",
    "2230", "2072", "2170", "1813", "1881", "2386", "2364", "1703", "704", "726",
    "1248", "2430", "1238", "1773", "434", "232", "1032", "613", "2435", "810",
    "1447", "224", "1324", "1272", "543", "522", "440", "1748", "604", "136",
    "753", "1050", "473", "2429", "719", "200", "573", "1536", "584", "1926",
    "1579", "1630", "1651", "826", "1680", "1216", "642", "2398", "1864", "127",
    "226", "1041", "1597", "1614", "1450", "1668", "2026", "1433", "1983", "686",
    "345", "1643", "1738", "2358", "1692", "695", "231", "716", "1902", "445",
    "544", "494", "507", "1064", "2428", "1212", "1853", "1806", "1675", "1370",
    "1502", "1467", "1944", "323", "156", "1787", "238", "1247", "1431", "566",
    "1679", "1828", "155", "1914", "1810", "1484", "1150", "1815", "338", "257",
    "1266", "1803", "1688", "208", "1894", "2177", "1677", "1699", "1831", "1893",
    "723", "168", "1463", "475", "273", "660", "699", "1739", "1665", "643",
    "1092", "1916", "433", "1841", "1298", "1418", "421", "2412", "1060", "2009",
    "1062", "1146", "1821", "1848", "2304", "1886", "1974", "1562", "1654", "2438",
    "2409", "442", "1800", "2312", "1320", "1737", "1115", "1964", "1514", "2066",
    "1817", "813", "1182", "1878", "365", "1628", "1329", "1887", "288", "1613",
    "1420", "1088", "397", "625", "1728", "2234", "1232", "2385", "731", "1550",
    "312", "1405", "2344", "1318", "1776", "1873", "745", "1822", "1861", "662",
    "1140", "256", "199", "1363", "1369", "756", "763", "1850", "412", "1977",
    "1843", "1221", "1476", "1569", "1868", "1819", "1057", "1895", "727", "1240",
    "644", "315", "333", "599", "1662", "2389", "2095", "1403", "2240", "328",
    "1871", "1872", "765", "2391", "1330", "1661", "1621", "730", "1331", "1442",
    "425", "1857", "1606", "386", "1860", "1093", "2188", "1991", "389", "1648",
    "1904", "1658", "797", "1267", "2174", "1911", "106", "1635", "1304", "280",
    "1325", "1154", "121", "1083", "621", "1701", "1585", "2384", "413", "393",
    "1693", "1949", "526", "1349", "1436", "266", "539", "1797", "1357", "1794",
    "1908", "1847", "673", "1849", "1928", "76", "1641", "2418", "1209", "1724",
    "1445", "2423", "1588", "769", "394", "1625", "1672", "1033", "1664", "1930",
    "2441", "1071", "777", "1889", "746", "249", "1068", "1774", "1924", "141",
    "1496", "1742", "1922", "634", "1025", "1427", "1633", "2226", "2110", "142",
    "258", "1609", "1901", "796", "1460", "1876", "1102", "1289", "1689", "1710",
    "290", "1457", "2076"
]

"""for pdf_link in pdf_links:
    # Navigate to the PDF link
    pdf_link = fr"https://maya.tase.co.il/company/{pdf_link}?view=finance"

    driver.get(pdf_link)

    # Wait for the page to load the PDF
    time.sleep(5)  # Adjust this based on your internet speed

    # Locate the save button and click it
    try:
        save_button = driver.find_element(By.CSS_SELECTOR, "button.tableBtn.listDrop.ng-scope")
        save_button.click()
        time.sleep(2)  # Allow time for the download to start
        # Wait for the CSV button to be clickable and then click it
        csv_button = driver.find_element(By.LINK_TEXT, "CSV")
        csv_button.click()

        time.sleep(2)  # Allow time for the download to start
    except Exception as e:
        print(f"Failed to click the save button on {pdf_link}: {e}")

# Close the browser
driver.quit()"""
import time
symbols_list = []

for id in pdf_links:
    link = fr"https://market.tase.co.il/en/market_data/company/{id}/about"
    driver.get(link)

    # Wait for the page to load the PDF
    time.sleep(0.5)  # Adjust this based on your internet speed

    # Locate the save button and click it
    try:
        symb = driver.find_elements(By.CSS_SELECTOR, "td.ColW_5")
        symbols_list.append(fr"{symb[1].text}.TA")
        print(fr"{symb[1].text}.TA")
    except Exception as e:
        print(f"Failed to click the save button on {link}: {e}")

# Close the browser
driver.quit()
# Creating a DataFrame
df = pd.DataFrame({
    'PDF Links': pdf_links,
    'Symbols': symbols_list
})

# Saving the DataFrame to an Excel file
with pd.ExcelWriter('symbols.xlsx') as writer:
    df.to_excel(writer, sheet_name='IDs and Symbols', index=False)
