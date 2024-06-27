# testing local version, should test installed package when done too
import pytest
from src import update_user_recs_urssi as update_user

""" def test_fun_function():
    assert update_user.fun_function(3) == 13 """


""" testdata = [
    (8),
    ("fail"),
]
# parametrize will provide the variables to test
@pytest.mark.parametrize("test_number", testdata)
def test_fun_function(test_number):
    assert update_user.fun_function(test_number) == 20
 """


def test_update_discipline():
    update_user.update_discipline()


""" def test_research_interests():
    update_user.research_interests()  """


""" recommendations = ["rec one", "rec two"]
def test_save_recommendations_to_json():
    update_user.save_recommendations_to_json(recommendations) """

""" def test_get_arxiv_rec():
    keywords = ['education', 'sociology', 'inequality', 'social mobility']
    update_user.get_arxiv_rec(keywords)  """


""" def test_get_osf_rec():
    update_user.get_osf_rec()  """
