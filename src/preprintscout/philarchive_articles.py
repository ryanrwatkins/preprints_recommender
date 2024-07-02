import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def get_philarchive():
    # Calculate yesterday's date in the format used by the website (adjust format as needed)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # URL of the website to scrape
    url = "https://philarchive.org/#selecteditems"

    # Send a GET request to the website
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the request was unsuccessful

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the first <div id="entries"> element
    entries_div = soup.find("div", id="entries")

    # Initialize an empty string to hold the formatted results
    philarchive_articles = []

    # Iterate through each <li class="entry"> within the <div id="entries">
    for entry in entries_div.find_all("li", class_="entry"):
        # Check if the entry has yesterday's date
        date_div = entry.find("div", class_="subtle", style="float:right")
        if date_div and date_div.text.strip() == yesterday:
            # Find the citation span
            citation_span = entry.find("span", class_="citation")
            if citation_span:
                # Extract the first <a> tag within the citation span for title and link
                title_link_a = entry.find("span", class_="citation").find(
                    "a", href=True
                )
                if title_link_a:
                    title = title_link_a.text.strip()
                    link = title_link_a["href"]
                    # Ensure the link is a complete URL
                    if not link.startswith("http"):
                        link = f"https://philarchive.org{link}"

                # Extract the abstract
                abstract_div = entry.find("div", class_="abstract")
                if abstract_div:
                    abstract = abstract_div.text.strip()

                # all are from philsophy
                oecd_discipline = "6.3 Philosophy ethics and religion"

                article_abstract = [
                    oecd_discipline,
                    link,
                    title,
                    abstract,
                ]

                philarchive_articles.append(article_abstract)

    print("number of philarchive article = " + str(len(philarchive_articles)))
    return philarchive_articles
