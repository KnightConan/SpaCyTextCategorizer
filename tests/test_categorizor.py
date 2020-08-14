import pytest
import pandas as pd
from spacytextcategorizer import SpacyTextCategorizer


@pytest.fixture
def stc():
    yield SpacyTextCategorizer('en_core_web_sm')


def test__transform(stc):
    y = ["A", "B", "AB", "O", "A", "O", "A", "B", "AB"]
    result = stc._transform(y)
    expected_result = [{'A': 1, 'AB': 0, 'B': 0, 'O': 0},
                       {'A': 0, 'AB': 0, 'B': 1, 'O': 0},
                       {'A': 0, 'AB': 1, 'B': 0, 'O': 0},
                       {'A': 0, 'AB': 0, 'B': 0, 'O': 1},
                       {'A': 1, 'AB': 0, 'B': 0, 'O': 0},
                       {'A': 0, 'AB': 0, 'B': 0, 'O': 1},
                       {'A': 1, 'AB': 0, 'B': 0, 'O': 0},
                       {'A': 0, 'AB': 0, 'B': 1, 'O': 0},
                       {'A': 0, 'AB': 1, 'B': 0, 'O': 0}]
    assert expected_result == result


def test__inverse_transform(stc):
    y_org = ["A", "B", "AB", "O", "A", "O", "A", "B", "AB"]
    stc._transform(y_org)
    y = pd.DataFrame([{'A': 1, 'AB': 0, 'B': 0, 'O': 0},
                      {'A': 0, 'AB': 0, 'B': 1, 'O': 0},
                      {'A': 0, 'AB': 1, 'B': 0, 'O': 0},
                      {'A': 0, 'AB': 0, 'B': 0, 'O': 1},
                      {'A': 1, 'AB': 0, 'B': 0, 'O': 0},
                      {'A': 0, 'AB': 0, 'B': 0, 'O': 1},
                      {'A': 1, 'AB': 0, 'B': 0, 'O': 0},
                      {'A': 0, 'AB': 0, 'B': 1, 'O': 0},
                      {'A': 0, 'AB': 1, 'B': 0, 'O': 0}])
    result = stc._inverse_transform(y.to_numpy())
    expected_result = ["A", "B", "AB", "O", "A", "O", "A", "B", "AB"]
    assert expected_result == result


@pytest.mark.parametrize("y, expected_classes",
                         [(["A", "B", "AB", "O", "A", "O", "A", "B", "AB"],
                           ["A", "AB", "B", "O"]),
                          (["A", "1", "5", "O", "A", "O", "A", "1", "5"],
                           ["1", "5", "A", "O"]),
                          (None, [])])
def test_classes_(stc, y, expected_classes):
    if y:
        stc._transform(y)
    print(type(stc.classes_), "======")
    assert stc.classes_ == expected_classes
