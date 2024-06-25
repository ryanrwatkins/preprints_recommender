#!/home/ryanrwatkins/Env/recommend/bin/python

import os
from os.path import exists
import logging
import django
import arxiv
import pandas as pd
#import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
import json
import google.generativeai as palm
import re
from datetime import datetime, timedelta, date
import pytz
import requests
import pw_file
from langdetect import detect
import time
from retry import retry

logging.basicConfig(filename='py_error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the Django environment to access django tools since this is a separate python file and would otherwise not have access
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "beta_recommend.settings")
django.setup()

# Now we can load the UserProfile
from rec_app.models import UserProfile


# HuggingFace API for getting embeddings -- somewhat based on https://huggingface.co/blog/getting-started-with-embeddings
def initiate_embedding(content):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"   #this model only embedds the first 256 words, but that is enough for our purposes and it is a small load which is better
    hf_token = pw_file.hf_api_key2
    content = content  
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    hf_headers = {"Authorization": f"Bearer {hf_token}"}
    # the initial API call loads the model, thus it has to use Retry decorator while the model loads, subsequent API calls will run faster
    @retry(tries=10, delay=10)
    def embedding(content):
        response = requests.post(api_url, headers=hf_headers, json={"inputs": content})
        embedding = response.json()
        if isinstance(embedding, list):
            return embedding
        elif list(embedding.keys())[0] == "error":
            raise RuntimeError(
                "The model is currently loading, please re-run the query."
                )   
        
    embedding = embedding(content)
    return embedding


# Connecting to BARD - PALM (in future could offer other LLM systems too)
def palm_llm(llm_prompt, temp, output_max, safety):
    palm.configure(api_key=pw_file.palm_api_key)
    # this gets the latest model for text generation
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name
    llm_completion = palm.generate_text(
        model=model,
        prompt=llm_prompt,
        temperature=temp,     
        max_output_tokens=output_max,
        safety_settings=[
            { "category": 1,  #1 to 6 
                "threshold": safety, # 1 is block most anything, 3 is high threshold, 4 is block none
            },
            {"category": 2,
                "threshold": safety,
            },
            {"category": 3,
                "threshold": safety,
            },
            {"category": 4,
                "threshold": safety,
            },
            {"category": 5,
                "threshold": safety,
            },
            {"category": 6,
                "threshold": safety,
            },
            ]
        
        )
    return llm_completion.result

# GENERATING KEYWORDS BASED ON USE PROFILE - OPTION TWO USING LLM
def get_keywords_llm(biography):
    print("finding keywords")
    global research_interests
    keywords_prompt = "Create a list of just four key words that describe the researcher's interest. Use an asterisk to start a new line for each key  word." + str(biography)
    temp = 0
    output_max = 800
    safety = 4
    keywords = palm_llm(keywords_prompt, temp, output_max, safety)

    pattern = "(?:\*.*)"
    keywords = re.findall(pattern,keywords)
    research_interests = [s.strip('* ') for s in keywords]
    print("found keywords")
    print(research_interests)
    return research_interests


# SELECTING PRIMARY DISCIPLINE FOR THE USER BAED ON PROFILE
def select_discipline(biography):
    print("getting discipline")
    global user_discipline
    global disciplines
    global user_embedding
    # Research areas and disciplines
    dissim_df = pd.read_csv("/var/www/html/beta.weshareresearch.com/public_html/beta_recommend/discipline_dissimilarity_matrix.csv")
    disciplines =  dissim_df["oecd_names"].tolist()
    user_embedding = initiate_embedding([str(biography)])
    discipline_embedding = initiate_embedding(disciplines)  

    similarities = cosine_similarity(user_embedding, discipline_embedding)  
    most_similar_idx = similarities.argmax()  # get the index of the highest similarity
    user_discipline = disciplines[most_similar_idx]
    print("got discipline - " + user_discipline)
    return user_discipline

# get arxiv articles based on key words (6 per keyword at this point)
def get_arxiv_rec(research_interests):
    print("getting arxiv articles")   
    global arxiv_articles
    arxiv_articles = []
    global number_days
    number_days = 1  

    utc=pytz.UTC
    now = utc.localize(datetime.now())
    # Enclose each word in quotes and join with " OR "
    search_query = ' OR '.join(f'"{keyword}"' for keyword in research_interests)
    #Uses double quote bcause it forces arXiv to match the keywords in the title, abstract or comments.  https://arxiv.org/multi?group=physics&%2Ffind=Search
    def get_arxiv():
        global number_days
        arxiv_tax_df = pd.read_csv("/var/www/html/beta.weshareresearch.com/public_html/beta_recommend/arxiv_taxonomy.csv")
        mapping_df = pd.read_csv("/var/www/html/beta.weshareresearch.com/public_html/beta_recommend/schema_mapping_cleaned.csv")

        MAX_RETRIES = 5
        RETRY_DELAY = 5 

        for keyword in research_interests:
            for attempt in range(MAX_RETRIES):
                try:
                    search = arxiv.Search(
                        query=keyword,
                        max_results=10,
                        sort_by=arxiv.SortCriterion.SubmittedDate
                    )
                    print(f"Processing keyword: {keyword}")
                    for result in search.results():
                        # go back day by day until you get recs. arXiv doesn't update on weekends so this is used
                        if now - timedelta(days=number_days) <= result.published <= now:
                            # add oecd discipline for later use in dissimilarity values
                            arxiv_group = arxiv_tax_df.loc[arxiv_tax_df["category_id"] == result.primary_category].iloc[0, 0]
                            oecd_discipline = mapping_df.loc[mapping_df["dc_arxiv_names"] == arxiv_group].iloc[0, 1]
                            article_abstract = [oecd_discipline, result.entry_id, result.title, result.summary]
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



      
    def retrieve_articles():
        global number_days
        MAX_DAYS_BACK = 10
        while len(arxiv_articles) < 10 and number_days <= MAX_DAYS_BACK:
            number_days += 1
            get_arxiv()
            
        if len(arxiv_articles) < 10:
            print("Couldn't retrieve at least 10 articles even after going back {} days.".format(MAX_DAYS_BACK))


    retrieve_articles()

    no_dups = []  # take out duplicate arXiv results and put in new list
    for elem in arxiv_articles:
        if elem not in no_dups:
            no_dups.append(elem)
    arxiv_articles = no_dups
    print("got arxiv articles " + str(len(arxiv_articles)))  
    return arxiv_articles

# get articles from OSF -- no keyword option, so we just get the last 24hrs (much fewer than arxiv so this works)
def get_osf_rec():
    global osf_articles
    print("getting OSF articles")
    osf_articles = []

    today = date.today()
    yesterday = str(today - timedelta(days = 1))

    mapping_df = pd.read_csv("/var/www/html/beta.weshareresearch.com/public_html/beta_recommend/schema_mapping_cleaned.csv")

    # get all preprints from yesterday
    osf_api = 'https://api.osf.io/v2/preprints/?filter%5Bdate_created%5D=' + yesterday + "&format=jsonapi"

    # api comes in 10 per page so we collect all the pages into one
    osf_api = requests.get(osf_api).json()
    all_articles = osf_api["data"]
    while osf_api["links"]["next"]:
        osf_api = requests.get(osf_api["links"]["next"]).json()
        all_articles.extend(osf_api["data"])



    for i in enumerate(all_articles):
        #first we may a list of subject areas we are not interested in based on fields are from https://www.bepress.com/wp-content/uploads/2016/12/bepress_Disciplines_taxonomy.pdf
        not_interested =   ["Psychiatry",]  #["Psychiatry", "Medicine and Health Sciences", "Life Sciences", "Mathematics", "Chemistry"]
        # then we make a list of all the subjects and sub-subjects that are listed for the preprint
        subjects_list = []
        for subjects in i[1]['attributes']['subjects'][0]:
            subjects_list.append(subjects['text'])
        # then we just want English preprints

        try:
            # we use a try function here since if langdetect can't find text to determine what language it may send an error (such as if there is no description given), and if that happens we tell it to continue on with the next iteration of the loop
            detect(i[1]['attributes']['description'])
            # if there is no error in the detect, then we can check the language
            if (detect(i[1]['attributes']['description']) == "en") and (detect(i[1]['attributes']['title']) == "en"):
                #if the description is in English, then we just want articles whose subjects are not listed in our not_interested list
                if not any(x in subjects_list for x in not_interested):
                    # add oecd discipline so that we can do dissimilirity value later
                    if len(i[1]['attributes']['subjects'][0]) == 2 :
                        article =  [i[1]['attributes']['subjects'][0][0]['text'] + ": " + i[1]['attributes']['subjects'][0][1]['text'], i[1]['links']['html'], i[1]['attributes']['title'], i[1]['attributes']['description']]
                        oecd_discipline = mapping_df.loc[mapping_df["dc_arxiv_names"] == article[0]].iloc[0, 1]
                    else:
                        article =  [i[1]['attributes']['subjects'][0][0]['text'], i[1]['links']['html'], i[1]['attributes']['title'], i[1]['attributes']['description']]
                        oecd_discipline = mapping_df.loc[mapping_df["dc_arxiv_names"] == article[0]].iloc[0, 1]
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


# remove duplicates for ranking function below just to be safe
def remove_duplicates(data):
    seen = set()
    undup_data = []

    for dic in data:
        key = tuple(dic['article'])  # Convert list to tuple
        if key in seen:
            continue

        undup_data.append(dic)
        seen.add(key)
    return undup_data

# have Bard add rationales for recommendations
def bard_rationale(recs_list, biography):
    print("adding rationales")
    for i in recs_list:
        # we limit the biography to the first 500 characters since BARD currently only takes 1000 tokens as input and we require space for the article abstract
        prompt_recs_rationale = "Read the following article abstract: " + str(i["article"][3]) + ".  Now provide a short rationale of 80 word or less for why the following researcher will want to read this article: " + str(biography)
        temp = 0.25
        output_max = 500
        safety = 4
        recs_rationale = palm_llm(prompt_recs_rationale, temp, output_max, safety)
        i['article'].append(recs_rationale)
    print("added rationales")
    return recs_list

# LLM only recommendations
def llm_ranked_article(biography, articles, source):
    global llm_results  
    print("getting LLM articles")
    shortened_articles = []
    if len(str(articles)) > 18000:      # for the long list of articles we need to limit the number of characters in the prompt to keep under the token limit
        for i in articles:
            data = [i[1], i[3][:500]]   # we are taking just the first 400 characters of the Abstracts so that our prompt doesn't get too long for PALM
            shortened_articles.append(data)
    else:
        for i in articles:
            data = [i[1], i[3][:800]]   # we are taking just the first 800 characters of the Abstracts so that our prompt doesn't get too long for PALM
            shortened_articles.append(data)

    llm_recs_prompt = "Forget prior results, start new. Review all, each and every, of the the following articles: " + str(shortened_articles) + ".  Then consider the following researcher:" + biography[:1000] + "Next order the articles from the list by how much it matches the researcher's interests, with the ones they will most want to read at the beginning of the list. Now Provide an asterisk bullet list with just and only the URLS for the top five articles that the researcher will want to read. Provide nothing else but bullet list of singe URLs without additional formatting."
    temp = 0
    output_max = 500
    safety = 4
    llm_recs_urls = palm_llm(llm_recs_prompt, temp, output_max, safety)

    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    llm_recs_urls = re.findall(url_pattern, str(llm_recs_urls))
    llm_recs = [sublist for sublist in articles  if sublist[1] in llm_recs_urls]

    llm_results = llm_recs[0:5]
    
    # add dissim_score to llm results
    llm_results_with_score = []
    for llm_result in llm_results:
        dissim_df = pd.read_csv("/var/www/html/beta.weshareresearch.com/public_html/beta_recommend/discipline_dissimilarity_matrix.csv")
        user_oecd = user_discipline  
        article_oecd = llm_result[0]  #discipline is the first value in the list for each article
        #then we match the user discipline to the article discipline in the matric to get the dissim_value
        dissim_value = dissim_df.loc[dissim_df["oecd_names"] == user_oecd][article_oecd].item()   #.round(3).item()
        if dissim_value == 0:
            dissim_value = .1
        if math.isnan(float(dissim_value)):   #  currently 6.4 Arts and music is returning nan for dissim_value so skipping it until I can figure out why
            continue
        llm_article_dict = {"article":llm_result, "dissim_value":dissim_value}
        llm_results_with_score.append(llm_article_dict)

    llm_results = bard_rationale(llm_results_with_score, biography)
    return (llm_results)


# This is for creating ranked by sentence-transformers
def ranked_articles(biography, articles, source, adjacent_value):
    global results
    global arxiv_filtered_results
    global osf_filetered_results
    print("ranking articles")
    results = []

    # then for each article from the arXiv/osf search results, create embedding
    for article in articles:
        article_embedding = initiate_embedding([article[3]])  
        cosine_scores = cosine_similarity(article_embedding, user_embedding)
        # return the just the cosine_score as an element in a list -- e.g  [.34343]
        score_list = cosine_scores.tolist()[0]
        # get the dissimilarity covariance for the article (i.e., distance from users home discipline)
        dissim_df = pd.read_csv("/var/www/html/beta.weshareresearch.com/public_html/beta_recommend/discipline_dissimilarity_matrix.csv")
        user_oecd = user_discipline  
        article_oecd = article[0]  #discipline is the first value in the list for each article
        #then we match the user discipline to the article discipline in the matric to get the dissim_value
        dissim_value = dissim_df.loc[dissim_df["oecd_names"] == user_oecd][article_oecd].item()   #.round(3).item()
        if dissim_value == 0:
            dissim_value = .1
        if math.isnan(float(dissim_value)):   #  currently 6.4 Arts and music is returning nan for dissim_value so skipping it until I can figure out why
            continue
        """
        user_weight = adjacent_value     #   equalition below is for 10 point scale -- on form we give them options for  5, 6,7,8  
       	weighted = cosine_scores * (user_weight * float(dissim_value) + (10 - user_weight) * (1 - float(dissim_value)))
        weighted = weighted.item()
        """
        article_dict = {"article":article, "score": score_list[0], "dissim_value":dissim_value}  # "weighted":weighted}
        results.append(article_dict)
    
  
    # We separate out arXiv from OSF at this piont because we sorted list from each for feeding into the Adj Recommendation, otherwise these could be the same script. There is likely a better way to do this.
    if source == "arxiv_":
        arxiv_filtered_results = [item for item in results if isinstance(item, dict)]
        arxiv_filtered_results = remove_duplicates(arxiv_filtered_results)
        # sort by score
        results = sorted(arxiv_filtered_results, key=lambda x: x['score'], reverse=True)
        # choose how many
        arxiv_results = results[0:5]
        # pass the rest for adj recs
        arxiv_filtered_results = results[5:]
        # add rationales
        arxiv_results = bard_rationale(arxiv_results, biography)
        print("got arxiv articles")    
        print("passing remaining arxiv articles " + str(len(arxiv_filtered_results)))
        return (arxiv_results, arxiv_filtered_results)



    if source == "osf_":
        osf_filtered_results = [item for item in results if isinstance(item, dict)]
        osf_filtered_results = remove_duplicates(osf_filtered_results)
        # sort by score
        results = sorted(osf_filtered_results, key=lambda x: x['score'], reverse=True)
        # choose how many
        osf_results = results[0:5]
        # pass the rest for adj recs
        osf_filtered_results = results[5:]
        # add rationales
        osf_results = bard_rationale(osf_results, biography)
        print("got osf LLM articles")  
        print("passing remaining osf articles " + str(len(osf_filtered_results)))        
        return (osf_results, osf_filtered_results)

"""    
def adjacent_recs(arxiv_filtered_results, osf_filtered_results, biography, adjacent_value):
    global merged_results
    print("getting adj recs")
    merged_results = arxiv_filtered_results + osf_filtered_results  # two lists
    merged_results = [item for item in merged_results if isinstance(item, dict)]   # now one list
    merged_results = sorted(merged_results, key=lambda x: x['weighted'], reverse=True)  #having False here puts them in ascending rather than descending
    # choose how many
    merged_results = merged_results[0:5]
    merged_results = bard_rationale(merged_results, biography)
    print("got adj recs")
    return merged_results
"""
# create adjacent recommendations and filter by user selected distance
def adjacent_recs_new(arxiv_filtered_results, osf_filtered_results, biography, adjacent_value):
    global merged_results
    print("getting adj recs")
    def get_desired_dissim_value(user_preference):
        dissim_value_mapping = {
            1: 0.01,
            2: 0.15,
            3: 0.4,
            4: 0.6
        }
        return dissim_value_mapping.get(user_preference, 0)


    merged_results = arxiv_filtered_results + osf_filtered_results  # two lists
    merged_results = [item for item in merged_results if isinstance(item, dict)]   # now one list
     # Get the dissim_value threshold based on user's preference
    desired_dissim = get_desired_dissim_value(adjacent_value)
    # Filter results based on the desired category's dissim_value
    filtered_results = [item for item in merged_results if float(item['dissim_value']) >= desired_dissim]
    # Sort the filtered results based on the cosine similarity score
    filtered_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True) 
    # Return the top 5 results
    filtered_results = filtered_results[0:5]
    filtered_results = bard_rationale(filtered_results, biography)
    print("got adj recs")
    return filtered_results


# command function for running the other functions in the right order and saving results for each user
def update_recommendations():
    """
        # this is for making changes to the DB for specific users if we have to
        if profile.user.pk == 2:    # pk gets use id
            profile.adjacent_value = 2   # then you select which value to change for the user
            profile.save()
    """

    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Loop through all user profiles and update the recs, this should work while we have few users but we will have to rethink for more users since each user update takes 3+ minutes
    for profile in UserProfile.objects.all():
        # get things ready      
        get_keywords_llm(profile.biography)

        user_discipline = select_discipline(profile.biography)
        profile.user_discipline_ai = user_discipline
        profile.save()

        get_arxiv_rec(research_interests)  # we use key words to query arxiv since it gets so submissions per day
        get_osf_rec() # osf doesn't have query in api but only gets ~30 to ~40 submissions per day so we get last 24 hours


        arxiv_llm_recs = llm_ranked_article(profile.biography, arxiv_articles, "arxiv_")
        profile.arxiv_llm_recs =  arxiv_llm_recs
        profile.save()

        osf_llm_recs = llm_ranked_article(profile.biography, osf_articles, "osf_")
        profile.osf_llm_recs = osf_llm_recs
        profile.save()

        arxiv_ranked, arxiv_filtered_results = ranked_articles(profile.biography, arxiv_articles, "arxiv_", profile.adjacent_value)
        profile.arxiv_recs = arxiv_ranked
        profile.save()

        osf_ranked, osf_filtered_results = ranked_articles(profile.biography, osf_articles, "osf_", profile.adjacent_value)
        profile.osf_recs = osf_ranked 
        profile.save()

        adj_ranked = adjacent_recs_new(arxiv_filtered_results, osf_filtered_results, profile.biography, profile.adjacent_value)
        profile.adj_recs = adj_ranked 
        
        profile.recs_updated_on = f"Updated on {current_datetime}" 
        profile.save()

    print("Recommendations updated successfully!")
    os._exit(0)

if __name__ == '__main__':
    try:
        update_recommendations()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
