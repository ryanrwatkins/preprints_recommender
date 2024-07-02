import os
import logging
import arxiv
from datetime import datetime, timedelta, date
import pytz
import requests
import time
import pandas as pd


def get_arxiv(research_interests):
    directory_path = os.path.dirname(os.path.abspath(__file__))
    taxonomy = os.path.join(directory_path, "resources/arxiv_taxonomy.csv")

    arxiv_tax_df = pd.read_csv(taxonomy)

    schema = os.path.join(directory_path, "resources/schema_mapping_cleaned.csv")
    utc = pytz.UTC
    now = utc.localize(datetime.now())
    number_days = 1
    mapping_df = pd.read_csv(schema)
    arxiv_articles = []

    MAX_RETRIES = 5
    RETRY_DELAY = 5
    MAX_DAYS_BACK = 10
    while len(arxiv_articles) < 10 and number_days <= MAX_DAYS_BACK:
        number_days += 1

        for keyword in research_interests:
            for attempt in range(MAX_RETRIES):
                try:
                    client = arxiv.Client()
                    arxiv_search = arxiv.Search(
                        query=keyword,
                        max_results=10,  # Make this a variable in user_profile file
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                    )
                    print(f"Processing keyword: {keyword}")
                    for result in client.results(arxiv_search):
                        # go back day by day until you get recs. arXiv doesn't update on weekends so this is used
                        if now - timedelta(days=number_days) <= result.published <= now:
                            # add oecd discipline for later use in dissimilarity values
                            arxiv_group = arxiv_tax_df.loc[
                                arxiv_tax_df["category_id"] == result.primary_category
                            ].iloc[0, 0]
                            oecd_discipline = mapping_df.loc[
                                mapping_df["dc_arxiv_names"] == arxiv_group
                            ].iloc[0, 1]
                            article_abstract = [
                                oecd_discipline,
                                result.entry_id,
                                result.title,
                                result.summary,
                            ]

                            arxiv_articles.append(article_abstract)

                except requests.exceptions.ConnectionError as e:
                    # Log the error, if you have a logging setup
                    logging.error(f"arXiv Attempt {attempt + 1} failed with error: {e}")
                    if attempt < MAX_RETRIES - 1:  # i.e. not the last attempt
                        logging.error(f"arXiv Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logging.error("arXiv Max retries reached. Exiting.")
                        os._exit(0)
                except Exception as e:
                    logging.error(f"Unexpected error for keyword '{keyword}': {e}")

    if len(arxiv_articles) < 10:
        print(
            "Couldn't retrieve at least 10 articles even after going back {} days.".format(
                MAX_DAYS_BACK
            )
        )

    no_dups = []  # take out duplicate arXiv results and put in new list
    for elem in arxiv_articles:
        if elem not in no_dups:
            no_dups.append(elem)
    arxiv_articles = no_dups
    print("got arxiv articles " + str(len(arxiv_articles)))
    return arxiv_articles
