import os
from datetime import datetime, timedelta, date
import requests
from langdetect import detect
import pandas as pd

global directory_path
directory_path = os.path.dirname(os.path.abspath(__file__))

global current_datetime
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_osf():
    global osf_articles
    print("getting OSF articles")
    osf_articles = []

    today = date.today()
    yesterday = str(today - timedelta(days=1))

    schema = os.path.join(directory_path, "resources/schema_mapping_cleaned.csv")
    mapping_df = pd.read_csv(schema)

    # get all preprints from yesterday
    osf_api = (
        "https://api.osf.io/v2/preprints/?filter%5Bdate_created%5D="
        + yesterday
        + "&format=jsonapi"
    )

    # api comes in 10 per page so we collect all the pages into one
    osf_api = requests.get(osf_api).json()
    all_articles = osf_api["data"]
    while osf_api["links"]["next"]:
        osf_api = requests.get(osf_api["links"]["next"]).json()
        all_articles.extend(osf_api["data"])

    for i in enumerate(all_articles):
        # first we may a list of subject areas we are not interested in based on fields are from https://www.bepress.com/wp-content/uploads/2016/12/bepress_Disciplines_taxonomy.pdf
        not_interested = [
            "Psychiatry",
        ]  # ["Psychiatry", "Medicine and Health Sciences", "Life Sciences", "Mathematics", "Chemistry"]
        # then we make a list of all the subjects and sub-subjects that are listed for the preprint
        subjects_list = []
        for subjects in i[1]["attributes"]["subjects"][0]:
            subjects_list.append(subjects["text"])
        # then we just want English preprints

        try:
            # we use a try function here since if langdetect can't find text to determine what language it may send an error (such as if there is no description given), and if that happens we tell it to continue on with the next iteration of the loop
            detect(i[1]["attributes"]["description"])
            # if there is no error in the detect, then we can check the language
            if (detect(i[1]["attributes"]["description"]) == "en") and (
                detect(i[1]["attributes"]["title"]) == "en"
            ):
                # if the description is in English, then we just want articles whose subjects are not listed in our not_interested list
                if not any(x in subjects_list for x in not_interested):
                    # add oecd discipline so that we can do dissimilirity value later
                    if len(i[1]["attributes"]["subjects"][0]) == 2:
                        article = [
                            i[1]["attributes"]["subjects"][0][0]["text"]
                            + ": "
                            + i[1]["attributes"]["subjects"][0][1]["text"],
                            i[1]["links"]["html"],
                            i[1]["attributes"]["title"],
                            i[1]["attributes"]["description"],
                        ]
                        oecd_discipline = mapping_df.loc[
                            mapping_df["dc_arxiv_names"] == article[0]
                        ].iloc[0, 1]
                    else:
                        article = [
                            i[1]["attributes"]["subjects"][0][0]["text"],
                            i[1]["links"]["html"],
                            i[1]["attributes"]["title"],
                            i[1]["attributes"]["description"],
                        ]
                        oecd_discipline = mapping_df.loc[
                            mapping_df["dc_arxiv_names"] == article[0]
                        ].iloc[0, 1]
                    article[0] = oecd_discipline
                    osf_articles.append(article)
        except:
            continue

    # remove duplicates
    no_dups = []
    for elem in osf_articles:
        if elem not in no_dups:
            no_dups.append(elem)
    osf_articles = no_dups
    print("got osf articles " + str(len(osf_articles)))
    return osf_articles
