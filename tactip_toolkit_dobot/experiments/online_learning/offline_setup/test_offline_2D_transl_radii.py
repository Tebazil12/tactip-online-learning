import offline_2D_transl_radii
import numpy as np


def test_load_data():
    all_data = offline_2D_transl_radii.load_data()
    assert len(all_data[0][20]) == 15
    assert len(all_data[0][0][1]) == 126
    assert (
        all_data[0][20][1][0][0]
    ) == 154.3017578125  # is having np.array ok for 2 highest levels? Probably should be list...
    assert type(all_data) is list
    assert type(all_data[0]) is np.ndarray  # in future needs to be list
    assert type(all_data[0][0]) is np.ndarray
    assert type(all_data[0][0][0]) is np.ndarray
    assert type(all_data[0][0][0][0]) is np.ndarray
    assert type(all_data[0][0][0][0][0]) is np.float64
