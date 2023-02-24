import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv

# open url
url = urlopen("https://www.trustpilot.com/review/dogwatch.com")
soup = BeautifulSoup(url,'html.parser')

# count reviews
count = soup.find("span", {"class": "typography_typography__QgicV typography_h2__wAVpO typography_weight-medium__UNMDK typography_fontstyle-normal__kHyN3 styles_reviewCount__wGBxK"}).get_text()
print("number of reviews is: ", count)
count = int(count.replace(",",""))

# create new csv file
csvfile = open('./ws.csv', 'w', newline='', encoding = 'utf-8')
writer = csv.writer(csvfile)
writer.writerow(['companyName','datePublished','ratingValue','reviewBody'])

# extract content and write in csv
i = 2
while i < count//20-1:
    for review in soup.findAll("article"):
        companyName = soup.title.get_text().split("Reviews")[0]
        companyName = companyName[:-1]
        datePublished = review.find("time")["datetime"]
        ratingValue = review.findChild("img")["alt"]
        reviewBody = review.find("p")
        if reviewBody is not None:
            reviewBody = reviewBody.get_text()
        writer.writerow([companyName,datePublished,ratingValue,reviewBody])
    print(i)
    page = urlopen('https://www.trustpilot.com/review/dogwatch.com?page='+str(i))
    soup = BeautifulSoup(page,'html.parser')
    i += 1

csvfile.close()

