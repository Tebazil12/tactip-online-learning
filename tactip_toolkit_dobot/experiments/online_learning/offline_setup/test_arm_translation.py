import load_json
import numpy as np


def test_whatever():
    all_data = load_json.load_data()
    ref_diffs_norm = load_json.whatever(all_data)
    assert ref_diffs_norm is not None


def test_load_data():
    all_data = load_json.load_data()

    # Check all_data is correct format

    assert type(all_data) is list
    # print(len(all_data))
    assert type(all_data[1]) is list
    # print(type(all_data[1]))
    # print(len(all_data[1]))
    assert type(all_data[1][1]) is list
    # print(type(all_data[1][1]))
    # print(len(all_data[1][1]))
    assert type(all_data[1][1][1]) is np.ndarray
    # print(type(all_data[1][1][1]))
    # print(len(all_data[1][1][1]))
    # print((all_data[1][1][1]))
    # print(type(all_data[1][1][1][1]))
    assert type(all_data[1][1][1][1]) is np.ndarray
    # print(type(all_data[1][1][1][1][1]))
    assert type(all_data[1][1][1][1][1]) is np.ndarray
    # print(type(all_data[1][1][1][1][1][1]))
    assert type(all_data[1][1][1][1][1][1]) is np.float64

    # all_data[1][1][1] = None

    # data[tap_number][frame][pin][xorydisp]

    for n_depths in all_data:
        for n_angles in n_depths:
            if any(elem is None for elem in n_angles):
                assert False
