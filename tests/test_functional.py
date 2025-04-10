import pytest
import pandas as pd

from src.utils.dir_utils import print_case_name


@print_case_name
def test_functional_from_text(predictor):
    text = "This is a news article about sports."
    label = predictor.predict(text)
    expected_label = "Sports"
    predictor.log_info("input text: {},expected label: {}, output label: {}".format(text, expected_label, label))
    assert label == "Sports"
@print_case_name
def test_functional_predict_from_csv(predictor):
    test_data = pd.read_csv("../dataset/data/functional_test_data.csv")
    for _, row in test_data.iterrows():
        text = row['text']
        expected_label = row['label']
        label = predictor.predict(text)
        predictor.log_info("input text: {},expected label: {}, output label: {}".format(text, expected_label, label))
        assert label == expected_label
