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
    State,
    plot_all_movements,
)

from tactip_toolkit_dobot.experiments.online_learning.contour_following_3d import make_meta
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp


def save_final_data(ex):
    # save the final set of data

    common.save_data(ex.all_raw_data, state.meta, name="all_data_final.json")
    common.save_data(ex.all_tap_positions, state.meta, name="all_positions_final.json")

def save_final_status():
    name, _ = os.path.split(state.meta["meta_file"])
    data = {
        "success": state.success,
        "useful": None,
        "name": name,
        "stimuli": state.meta["stimuli_name"],
        "comment": "",
    }

    with open(os.path.join(state.meta["home_dir"], "experiment_logs"), "a") as myfile:
        json.dump(data, myfile)
        myfile.write("\n")


def plot_profiles_flat(results, meta, ref_tap):
    # get dissim profile
    for line in results:

        dissim_profile = dp.calc_dissims(
            np.array(line), ref_tap
        )  # taps needs casting as the eulicd distance can be done on all at once (instead of looping list)
        plt.plot(meta["line_range"], dissim_profile)

    part_path, _ = os.path.split(meta["meta_file"])
    full_path_png = os.path.join(meta["home_dir"], part_path, "dissim_profiles.png")
    full_path_svg = os.path.join(meta["home_dir"], part_path, "dissim_profiles.svg")
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()
    plt.clf()

np.set_printoptions(precision=2, suppress=True)


def main(ex, model, meta):

    with common.make_robot() as ex.robot, common.make_sensor(meta) as ex.sensor:
        # print("now clearing")
        # results = ex.robot.sync_robot.clearAlarms()

        print("initing robot")
        common.init_robot(ex.robot, meta, do_homing=True)

        print("Main code...")

        ex.collect_neutral_tap(meta)

        common.go_home(ex.robot, meta)

        if meta["collect_ref_tap"] is True:
            ref_tap = ex.collect_ref_tap(meta)

        results =[]

        # for height in range(-1, 2, 1):
        for height in meta["height_range"]:
            # for angle in range(-45, 46, 5):
            for angle in meta["angle_range"]:

                results.append(ex.collect_line(
                    [0, 0], np.deg2rad(angle), meta, height=height
                ))

                # print("collected:")
                # print(results)
                #
                # print("raw data")
                # print(ex.all_raw_data)

                time.sleep(1) # to give tip time to return to neutral

        common.go_home(ex.robot, meta)
    # save data
    save_final_data(ex)
    plot_all_movements(ex, meta)
    plot_profiles_flat(results, meta, ref_tap)
    save_final_status()

    print("Done, exiting")


# state = State(
#         meta=make_meta(file_name="collect_dataset_3d.py", stimuli_name="105mm-circle")
#     )

if __name__ == "__main__":
    extra_dict = {
        # range(-1, 2, 1)
        "height_range": np.array(np.arange(-1, 2.5001, 0.5)).tolist(),
        # range(-45, 46, 5)
        # "angle_range": np.array(range(-45, 46, 5)).tolist(),
        "angle_range": np.array(range(-15, 16, 5)).tolist(),
        # "line_range": np.arange(-10, 11, 1).tolist(),
        "line_range": np.arange(-10, 11, 2).tolist(),
        "ref_location": [0, 0, 0],
        "comments": "bug should be fixed now"
    }

    state = State(
        meta=make_meta(
            file_name="collect_dataset_3d.py",
            # stimuli_name="105mm-circle",
            stimuli_name="tilt-05deg-up-offline",
            extra_dict=extra_dict,
        )
    )

    main(state.ex, state.model, state.meta)
