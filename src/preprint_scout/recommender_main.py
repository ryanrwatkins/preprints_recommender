import os
import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math
import json
import re
from datetime import datetime


# the first relative imports are for pytest since it uses a different directory, and if not pytest then it uses the regular
try:
    from .user_profile import user_profile
    from .arxiv_articles import get_arxiv
    from .osf_articles import get_osf
    from .philarchive_articles import get_philarchive
    from .embedding import initiate_embedding
    from .llms import palm_llm
except ImportError:
    from user_profile import user_profile
    from arxiv_articles import get_arxiv
    from osf_articles import get_osf
    from philarchive_articles import get_philarchive
    from embedding import initiate_embedding
    from .llms import palm_llm

logging.basicConfig(
    filename="py_error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

global directory_path
directory_path = os.path.dirname(os.path.abspath(__file__))

global current_datetime
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

""" # this is for playing with pytest
def fun_function(value):
    x = value + 12
    return x
 """


def get_keywords_llm(biography):
    """Uses the user biography to identify key words that will be used in arxiv search"""
    print("finding keywords")
    global research_interests
    keywords_prompt = (
        "Create a list of just four key words that describe the researcher's interest. Use an asterisk to start a new line for each key  word."
        + str(biography)
    )
    temp = 0
    output_max = 800
    safety = 4
    keywords = palm_llm(keywords_prompt, temp, output_max, safety)

    pattern = "(?:\*.*)"
    keywords = re.findall(pattern, keywords)
    research_interests = [s.strip("* ") for s in keywords]
    print("found keywords")
    print(research_interests)
    return research_interests


def select_discipline(biography):
    """Uses the user biography to select an appropriate discipline, which is later used for adjacent recommendations"""
    print("getting discipline")
    global user_discipline
    global disciplines
    global user_embedding
    # Research areas and disciplines
    """ matrix = working_directory + "/src/discipline_dissimilarity_matrix.csv" """
    """ matrix = working_directory + "/discipline_dissimilarity_matrix.csv" """
    matrix = os.path.join(
        directory_path, "resources/discipline_dissimilarity_matrix.csv"
    )
    dissim_df = pd.read_csv(matrix)
    disciplines = dissim_df["oecd_names"].tolist()
    print("starting to get embeddings")
    user_embedding = initiate_embedding([str(biography)])
    print("getting user_embedding")
    discipline_embedding = initiate_embedding(disciplines)
    print("getting discipline_embedding")
    similarities = cosine_similarity(user_embedding, discipline_embedding)
    most_similar_idx = similarities.argmax()  # get the index of the highest similarity
    user_discipline = disciplines[most_similar_idx]
    print("user discipline - " + user_discipline)
    return user_discipline


def remove_duplicates(data):
    """remove duplicates for ranking function below just to be safe"""
    seen = set()
    undup_data = []

    for dic in data:
        key = tuple(dic["article"])  # Convert list to tuple
        if key in seen:
            continue

        undup_data.append(dic)
        seen.add(key)
    return undup_data


def add_rationale(recs_list, biography):
    """Uses the LLM to write rationales for each recommendation."""
    print("adding rationales")
    for i in recs_list:
        # we limit the biography to the first 500 characters since BARD currently only takes 1000 tokens as input and we require space for the article abstract
        prompt_recs_rationale = (
            "Read the following article abstract: "
            + str(i["article"][3])
            + ".  Now provide a short rationale of 80 word or less for why the following researcher will want to read this article: "
            + str(biography)
        )
        temp = 0.25
        output_max = 500
        safety = 4
        recs_rationale = palm_llm(prompt_recs_rationale, temp, output_max, safety)
        i["article"].append(recs_rationale)
    print("added rationales")
    return recs_list


def llm_ranked_article(biography, articles, source):
    """This uses the LLM to determine which articles from the list should be recommended for the user biography"""
    global llm_results
    print("getting LLM ranking of articles")
    shortened_articles = []
    if (
        len(str(articles)) > 18000
    ):  # for the long list of articles we need to limit the number of characters in the prompt to keep under the token limit
        for i in articles:
            data = [
                i[1],
                i[3][:500],
            ]  # we are taking just the first 400 characters of the Abstracts so that our prompt doesn't get too long for PALM
            shortened_articles.append(data)
    else:
        for i in articles:
            data = [
                i[1],
                i[3][:800],
            ]  # we are taking just the first 800 characters of the Abstracts so that our prompt doesn't get too long for PALM
            shortened_articles.append(data)

    llm_recs_prompt = (
        "Forget prior results, start new. Review all, each and every, of the the following articles: "
        + str(shortened_articles)
        + ".  Then consider the following researcher:"
        + biography[:1000]
        + "Next order the articles from the list by how much it matches the researcher's interests, with the ones they will most want to read at the beginning of the list. Now Provide an asterisk bullet list with just and only the URLS for the top five articles that the researcher will want to read. Provide nothing else but bullet list of singe URLs without additional formatting."
    )
    temp = 0
    output_max = 500
    safety = 4
    llm_recs_urls = palm_llm(llm_recs_prompt, temp, output_max, safety)

    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    llm_recs_urls = re.findall(url_pattern, str(llm_recs_urls))
    llm_recs = [sublist for sublist in articles if sublist[1] in llm_recs_urls]

    llm_results = llm_recs[0:5]

    # add dissim_score to llm results
    llm_results_with_score = []
    for llm_result in llm_results:
        """matrix = working_directory + "/src/discipline_dissimilarity_matrix.csv" """
        """ matrix = working_directory + "/discipline_dissimilarity_matrix.csv" """
        matrix = os.path.join(
            directory_path, "resources/discipline_dissimilarity_matrix.csv"
        )
        dissim_df = pd.read_csv(matrix)
        user_oecd = user_discipline
        article_oecd = llm_result[
            0
        ]  # discipline is the first value in the list for each article
        # then we match the user discipline to the article discipline in the matric to get the dissim_value
        dissim_value = dissim_df.loc[dissim_df["oecd_names"] == user_oecd][
            article_oecd
        ].item()  # .round(3).item()
        if dissim_value == 0:
            dissim_value = 0.1
        if math.isnan(
            float(dissim_value)
        ):  #  currently 6.4 Arts and music is returning nan for dissim_value so skipping it until I can figure out why
            continue
        llm_article_dict = {"article": llm_result, "dissim_value": dissim_value}
        llm_results_with_score.append(llm_article_dict)

    llm_results = add_rationale(llm_results_with_score, biography)
    return llm_results


def cosine_ranked_articles(biography, articles, source, adjacent_value):
    """For creating ranked recs using cosine-similarity by sentence-transformers on huggingface"""
    global results
    global arxiv_filtered_results
    global osf_filetered_results
    global philarchive_filetered_results
    print("ranking articles")
    results = []

    # then for each article from the arXiv/osf search results, create embedding
    for article in articles:
        article_embedding = initiate_embedding([article[3]])
        cosine_scores = cosine_similarity(article_embedding, user_embedding)
        # return the just the cosine_score as an element in a list -- e.g  [.34343]
        score_list = cosine_scores.tolist()[0]
        # get the dissimilarity covariance for the article (i.e., distance from users home discipline)
        """ matrix = working_directory + "/src/discipline_dissimilarity_matrix.csv"  """
        """ matrix = working_directory + "/discipline_dissimilarity_matrix.csv" """
        matrix = os.path.join(
            directory_path, "resources/discipline_dissimilarity_matrix.csv"
        )
        dissim_df = pd.read_csv(matrix)
        user_oecd = user_discipline
        article_oecd = article[
            0
        ]  # discipline is the first value in the list for each article
        # then we match the user discipline to the article discipline in the matric to get the dissim_value
        dissim_value = dissim_df.loc[dissim_df["oecd_names"] == user_oecd][
            article_oecd
        ].item()  # .round(3).item()
        if dissim_value == 0:
            dissim_value = 0.1
        if math.isnan(
            float(dissim_value)
        ):  #  currently 6.4 Arts and music is returning nan for dissim_value so skipping it until I can figure out why
            continue
        """
        user_weight = adjacent_value     #   equalition below is for 10 point scale -- on form we give them options for  5, 6,7,8
       	weighted = cosine_scores * (user_weight * float(dissim_value) + (10 - user_weight) * (1 - float(dissim_value)))
        weighted = weighted.item()
        """
        article_dict = {
            "article": article,
            "score": score_list[0],
            "dissim_value": dissim_value,
        }  # "weighted":weighted}
        results.append(article_dict)

    # We separate out arXiv from OSF at this piont because we sorted list from each for feeding into the Adj Recommendation, otherwise these could be the same script. There is likely a better way to do this.
    if source == "arxiv_":
        arxiv_filtered_results = [item for item in results if isinstance(item, dict)]
        arxiv_filtered_results = remove_duplicates(arxiv_filtered_results)
        # sort by score
        results = sorted(arxiv_filtered_results, key=lambda x: x["score"], reverse=True)
        # choose how many
        arxiv_results = results[0:5]
        # pass the rest for adj recs
        arxiv_filtered_results = results[5:]
        # add rationales
        arxiv_results = add_rationale(arxiv_results, biography)
        print("got arxiv articles")
        print("passing remaining arxiv articles " + str(len(arxiv_filtered_results)))
        return (arxiv_results, arxiv_filtered_results)

    if source == "osf_":
        osf_filtered_results = [item for item in results if isinstance(item, dict)]
        osf_filtered_results = remove_duplicates(osf_filtered_results)
        # sort by score
        results = sorted(osf_filtered_results, key=lambda x: x["score"], reverse=True)
        # choose how many
        osf_results = results[0:5]
        # pass the rest for adj recs
        osf_filtered_results = results[5:]
        # add rationales
        osf_results = add_rationale(osf_results, biography)
        print("got osf LLM articles")
        print("passing remaining osf articles " + str(len(osf_filtered_results)))
        return (osf_results, osf_filtered_results)

    if source == "philarchive_":
        philarchive_filtered_results = [
            item for item in results if isinstance(item, dict)
        ]
        philarchive_filtered_results = remove_duplicates(philarchive_filtered_results)
        # sort by score
        results = sorted(
            philarchive_filtered_results, key=lambda x: x["score"], reverse=True
        )
        # choose how many
        philarchive_results = results[0:5]
        # pass the rest for adj recs
        philarchive_filtered_results = results[5:]
        # add rationales
        philarchive_results = add_rationale(philarchive_results, biography)
        print("got philarchive LLM articles")
        print(
            "passing remaining philarchive articles "
            + str(len(philarchive_filtered_results))
        )
        return (philarchive_results, philarchive_filtered_results)


def adjacent_recs(
    arxiv_filtered_results,
    osf_filtered_results,
    philarchive_filtered_results,
    biography,
    adjacent_value,
):
    """Uses the adjacent fields value from the user to recommend preprints that are disciplinarily
    further from their home discipline"""
    global merged_results
    print("getting adj recs")

    def get_desired_dissim_value(user_preference):
        dissim_value_mapping = {1: 0.01, 2: 0.15, 3: 0.4, 4: 0.6}
        return dissim_value_mapping.get(user_preference, 0)

    merged_results = (
        arxiv_filtered_results + osf_filtered_results + philarchive_filtered_results
    )
    merged_results = [
        item for item in merged_results if isinstance(item, dict)
    ]  # now one list
    # Get the dissim_value threshold based on user's preference
    desired_dissim = get_desired_dissim_value(adjacent_value)
    # Filter results based on the desired category's dissim_value
    filtered_results = [
        item for item in merged_results if float(item["dissim_value"]) >= desired_dissim
    ]
    # Sort the filtered results based on the cosine similarity score
    filtered_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)
    # Return the top 5 results
    filtered_results = filtered_results[0:5]
    filtered_results = add_rationale(filtered_results, biography)
    print("got adj recs")
    return filtered_results


def save_recommendations_to_json(recommendations, directory="src/outputs"):
    """file_path = working_directory + f"/src/outputs/recommendations_{current_datetime}.json" """
    """ file_path = working_directory + f"/outputs/recommendations_{current_datetime}.json" """
    file_path = os.path.join(
        directory_path, f"outputs/recommendations_{current_datetime}.json"
    )
    with open(file_path, "w") as json_file:
        json.dump(recommendations, json_file, indent=4)
    print(f"Recommendations saved to {file_path}")


def research_interests():
    """takes profile and identifies keywords that will be used in arxiv search"""
    research_interests_keywords = get_keywords_llm(user_profile.biography)
    user_profile.keywords = research_interests_keywords
    user_profile.save()


def update_discipline():
    """Take the user bio and then determines what OECD discipine they fall withing"""
    user_discipline = select_discipline(user_profile.biography)
    user_profile.discipline = user_discipline
    user_profile.save()


# command function for running the other functions in the right order and saving results for each user
def update_recommendations():
    """Creates the five different types of recommendations and saves to json"""
    research_interests()
    update_discipline()

    print("getting arxiv_llm_recs")
    arxiv_articles = get_arxiv(research_interests)
    arxiv_llm_recs = llm_ranked_article(
        user_profile.biography, arxiv_articles, "arxiv_"
    )

    print("getting osf_llm_recs")
    osf_articles = get_osf()
    osf_llm_recs = llm_ranked_article(user_profile.biography, osf_articles, "osf_")

    print("getting osf_llm_recs")
    philarchive_articles = get_philarchive()
    philarchive_llm_recs = llm_ranked_article(
        user_profile.biography, philarchive_articles, "philarchive_"
    )

    arxiv_ranked, arxiv_filtered_results = cosine_ranked_articles(
        user_profile.biography, arxiv_articles, "arxiv_", user_profile.adjacent_value
    )

    osf_ranked, osf_filtered_results = cosine_ranked_articles(
        user_profile.biography, osf_articles, "osf_", user_profile.adjacent_value
    )

    philarchive_ranked, philarchive_filtered_results = cosine_ranked_articles(
        user_profile.biography,
        philarchive_articles,
        "philarchive_",
        user_profile.adjacent_value,
    )

    adj_ranked = adjacent_recs(
        arxiv_filtered_results,
        osf_filtered_results,
        philarchive_filtered_results,
        user_profile.biography,
        user_profile.adjacent_value,
    )

    recommendations = {
        "arxiv_llm_recs": arxiv_llm_recs,
        "osf_llm_recs": osf_llm_recs,
        "philarchive_llm_recs": philarchive_llm_recs,
        "arxiv_cosine_ranked": arxiv_ranked,
        "osf_cosine_ranked": osf_ranked,
        "philarchive_cosine_ranked": philarchive_ranked,
        "adj_ranked": adj_ranked,
    }

    # save_recommendations_to_json(recommendations) # Creates an archive of past recommendations in /outputs
    print("Recommendations updated successfully!")
    return recommendations
    # os._exit(0)


if __name__ == "__main__":
    try:
        update_recommendations()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
