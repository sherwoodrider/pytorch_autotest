import pytest

from src.utils.dir_utils import print_case_name



@pytest.mark.parametrize("text",[
    "!!! @@@ ##$$%%^^&&*()",
    "", "a" * 50,"Some random text with no meaning.",
])
@print_case_name
def test_predict_with_noisy_data(predictor, text):
    label = predictor.predict(text)
    predictor.log_info("input text: {}, output label: {}".format(text, label))
    assert label in ["World", "Sports", "Business", "Sci/Tech"]
