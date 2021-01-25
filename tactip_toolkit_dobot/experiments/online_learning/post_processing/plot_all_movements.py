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
    State
)


# np.set_printoptions(precision=2)#, suppress=True)


def main(ex,meta):





    ex.all_tap_positions = common.load_data(data_home + current_experiment + "all_positions_final.json")
    ex.all_tap_positions = np.array(ex.all_tap_positions)

    ex.line_locations = common.load_data(data_home + current_experiment + "location_line_002.json")
    ex.line_locations = np.array([ex.line_locations])

    print(ex.line_locations)
    print(type(ex.line_locations))
    print(np.shape(ex.line_locations))

    ex.edge_locations = common.load_data(data_home + current_experiment + "all_edge_locs_final.json")
    ex.edge_locations = np.array(ex.edge_locations)

    plot_all_movements(ex,meta)

if __name__ == "__main__":
    state = State()
    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    # current_experiment = "contour_following_2d_01m-19d_10h47m37s/"
    # current_experiment = "contour_following_2d_01m-18d_17h41m48s/"
    current_experiment = "contour_following_2d_01m-22d_14h58m05s/"

    state.meta = common.load_data(data_home + current_experiment + "meta.json")

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(state.ex,state.meta)
