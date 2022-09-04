"""Test mdx_e2c."""
import pytest
from tinybee.mdx_e2c import mdx_e2c


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("test", "测试"),
        ("tEst", "测试"),
        (" tEst !'", "测试"),  # strip ' ' + string.punctuation
        (" tEst -!'", "tEst -"),  # keep '-', return "tEst -"
        ("我", "我"),  # return original
        ("ttest", "ttest"),  # return original
        ("Ttest", "Ttest"),
        # ("", "a")  # ought to fail
        pytest.param("", "a", marks=pytest.mark.xfail),
    ]
)
def test_mdx_e2c(test_input, expected):
    """Test mdx_e2c."""
    assert expected in mdx_e2c(test_input)
    