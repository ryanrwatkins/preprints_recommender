# testing local version, should test installed package when done too
import pytest
import os

# Must in PreprintScout (main folder) and then run >>> pytest -s


import preprintscout

from src.preprintscout import arxiv_articles as arxiv_articles
from src.preprintscout import osf_articles as osf_articles
from src.preprintscout import philarchive_articles as philarchive_articles

# from src.preprint_scout import api_keys
# from src.preprint_scout.user_profile import UserProfile, user_profile

""" def test_fun_function():
    assert recommender_main.fun_function(3) == 13 """


""" testdata = [
    (8),
    ("fail"),
]
# parametrize will provide the variables to test
@pytest.mark.parametrize("test_number", testdata)
def test_fun_function(test_number):
    assert recommender_main.fun_function(test_number) == 20
 """


""" def test_update_discipline():
    recommender_main.update_discipline() """

""" def test_get_arxiv():
    interests = ['AI', 'math', 'machine learning']
    arxiv_articles.get_arxiv(interests) """

""" def test_get_osf():
    osf_articles.get_osf() """

""" def test_get_philarchive():
    philarchive_articles.get_philarchive() """

""" def test_research_interests():
    recommender_main.research_interests()  """


""" recommendations = ["rec one", "rec two"]
def test_save_recommendations_to_json():
    recommender_main.save_recommendations_to_json(recommendations) """

""" def test_get_arxiv_rec():
    keywords = ['education', 'sociology', 'inequality', 'social mobility']
    recommender_main.get_arxiv_rec(keywords)

 """
""" def test_get_osf_rec():
    recommender_main.get_osf_rec()  """


""" def test_get_osf_rec():
    recommender_main.update_recommendations() """

biography = "I am a engineer who studies artificial intelligence in electric vehicles."
huggingface_api_key = "xxxxx"  # api_keys.hf_api_key
google_api_key = "xxxxxx"  # api_keys.palm_api_key


def test_get_osf_rec():
    preprintscout.recommendation_main(
        biography,
        huggingface_api_key=huggingface_api_key,
        google_api_key=google_api_key,
        interdisciplinary="3",
        output_path="/Users/rwatkins_1/Python projects/preprints",
    )


""" def test_get_osf_rec():
    recommendations = ["sample", "sample"]
    recommender_main.save_recommendations_to_json(
        recommendations, "/Users/rwatkins_1/Python projects/preprints"
    ) """
