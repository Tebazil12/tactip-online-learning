import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
    plot_all_movements,
    State,
    next_sensor_placement
)


# np.set_printoptions(precision=2)#, suppress=True)
def load_data(ex):
    ex.all_tap_positions = common.load_data(data_home + current_experiment + "all_positions_final.json")
    ex.all_tap_positions = np.array(ex.all_tap_positions)

    ex.line_locations = common.load_data(data_home + current_experiment + "location_line_002.json")
    ex.line_locations = np.array([ex.line_locations])

    print(ex.line_locations)
    print(type(ex.line_locations))
    print(np.shape(ex.line_locations))

    ex.edge_locations = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
    ex.edge_locations = np.array(ex.edge_locations)


def main(ex,meta):
    print("edge locs:")
    print(ex.edge_locations)
    ex.edge_locations = [[0,2]]
    print(ex.robot)

    new_orient, new_location = next_sensor_placement(ex,meta)

    print(f"orient: {new_orient} loc: {new_location}")

if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    # current_experiment = "contour_following_2d_01m-19d_10h47m37s/"
    # current_experiment = "contour_following_2d_01m-18d_17h41m48s/"
    current_experiment = "contour_following_2d_01m-22d_14h58m05s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    load_data(state.ex)
    main(state.ex,state.meta)
