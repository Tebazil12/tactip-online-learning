"""
Author: Elizabeth A. Stone (lizzie.stone@brl.ac.uk)

Use:
A minimum working example of the tactile core code running with a
dobot magician robot arm.

This program sets up the robot arm and tactile sensor, saving meta data about
the current settings, and then does a single tap on an object (assuming there
is an object just below the location specified by work_frame).  The tap data is
keypoints (pin locations) and is saved to a data file and shown on a scatter
graph. The robot then returns to the home location, and the program closes.

The main variables for this program are stored in meta, as configured in
make_meta(), and these may need tweaking for your specific setup.

"""


import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import atexit

import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
    State,
    plot_all_movements,
)
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp


def save_final_data(ex):
    # save the final set of data

    common.save_data(ex.all_raw_data, state.meta, name="all_data_final.json")
    common.save_data(ex.all_tap_positions, state.meta, name="all_positions_final.json")


def plot_profiles(results, meta, ref_tap):
    # get dissim profile
    dissim_profile = dp.calc_dissims(
        np.array(results), ref_tap
    )  # taps needs casting as the eulicd distance can be done on all at once (instead of looping list)
    plt.plot(meta["line_range"], dissim_profile)
    plt.show()

np.set_printoptions(precision=2, suppress=True)


def main(ex, model, meta):

    with common.make_robot() as ex.robot, common.make_sensor(meta) as ex.sensor:
        # print("now clearing")
        # results = ex.robot.sync_robot.clearAlarms()

        print("initing robot")
        common.init_robot(ex.robot, meta, do_homing=False)

        print("Main code...")

        ex.collect_neutral_tap(meta)

        if meta["collect_ref_tap"] is True:
            ref_tap = ex.collect_ref_tap(meta)

        # for height in range(-1, 2, 1):
        for height in meta["height_range"]:
            # for angle in range(-45, 46, 5):
            for angle in meta["angle_range"]:

                results = ex.collect_line(
                    [0, 0], np.deg2rad(angle), meta, height=height
                )

                # print("collected:")
                # print(results)
                #
                # print("raw data")
                # print(ex.all_raw_data)

        common.go_home(ex.robot, meta)
    # save data
    save_final_data(ex)
    plot_all_movements(ex, meta)
    plot_profiles(results, meta, ref_tap)

    print("Done, exiting")


# state = State(
#         meta=make_meta(file_name="collect_dataset_3d.py", stimuli_name="105mm-circle")
#     )

if __name__ == "__main__":
    extra_dict = {
        # range(-1, 2, 1)
        "height_range": np.array(range(0, 1, 1)).tolist(),
        # range(-45, 46, 5)
        "angle_range": np.array(range(0, 1, 45)).tolist(),
        "ref_location": [0, 0, 0]
    }

    state = State(
        meta=make_meta(
            file_name="collect_dataset_3d.py",
            stimuli_name="105mm-circle",
            extra_dict=extra_dict,
        )
    )

    main(state.ex, state.model, state.meta)
