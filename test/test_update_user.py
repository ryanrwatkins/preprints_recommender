# testing local version, should test installed package when done too
import pytest
from src import update_user_recs_urssi as update_user

""" def test_fun_function():
    update_user.fun_function == 5 """


testdata = [
    (8),
    (5),
]
# parametrize will provide the variables to test
@pytest.mark.parametrize("test_number", testdata)
def test_fun_function(test_number):
    update_user.fun_function == test_number
