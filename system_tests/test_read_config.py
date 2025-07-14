
import pytest

from src.config_utils import read_config_scopes

def test_read_config_scopes(step_callback):
    scopes = read_config_scopes()

    step_callback(step_output=scopes)


