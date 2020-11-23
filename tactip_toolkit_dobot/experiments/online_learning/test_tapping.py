"""
Author: Elizabeth A. Stone (lizzie.stone@brl.ac.uk)

Use:

"""


import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common

def make_meta():
    """
    Make dictionary of all meta data about the current experiment, and
    save to json file.
    """

    # name for this experiment run folder and meta file (will be added to end
    # of "home_dir" for full path name)
    meta_file = os.path.join(
        "min_example",
        os.path.basename(__file__)[:-3] + "_" + time.strftime("%mm-%dd_%Hh%Mm%Ss"),
        "meta.json",
    )
    data_dir = os.path.dirname(meta_file)

    meta = {
        # ~~~~~~~~~ Paths ~~~~~~~~~#
        # directory for storing data on linux
        "home_dir": os.path.join(
            "/home/lizzie/git/tactip_toolkit_dobot/data", "TacTip_dobot"
        ),  # TODO, THIS WILL BREAK ON OTHER MACHINES
        # # directory for storing data on windows
        # "home_dir": os.path.join(
        #     os.environ["DATAPATH"], "TacTip_dobot"
        # ),
        "meta_file": meta_file,
        # "image_dir": None,
        # "image_df_file": None,
        # "ip": None,
        # ~~~~~~~~~ Robot movements ~~~~~~~~~#
        "robot_tcp": [0, 0, 150, 0, 0, 0],  # tool center point in mm - change for different sensors
        "base_frame": [0, 0, 0, 0, 0, 0],  # see dobot manual for location
        "home_pose": [170, 0, -150, 0, 0, 0],  # choose a safe "resting" pose (in base frame)
        "work_frame": [173, -5, -189, 0, 0, 0],  # experiment specific start point (in base frame)
        "linear_speed": 100,
        "angular_speed": 100,
        "tap_move": [[0, 0, -3, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        # "poses_rng": None,
        # "obj_poses": None,
        # "num_poses": None,
        # ~~~~~~~~~ CV Settings ~~~~~~~~~#
        "min_threshold": 60,
        "max_threshold": 320,
        "filter_by_color": True,
        "blob_color": 255,
        "filter_by_area": True,
        "min_area": 100,
        "max_area": 400,
        "filter_by_circularity": True,
        "min_circularity": 0.3,
        "filter_by_inertia": True,
        "min_inertia_ratio": 0.22,
        "filter_by_convexity": True,
        "min_convexity": 0.61,
        "nntracker_threshold": 20,
        "kpview_colour": (0, 255, 0),
        # ~~~~~~~~~ Camera Settings ~~~~~~~~~#
        "exposure": None,  # tacFoot camera doesn't support exposure
        "brightness": 255,
        "contrast": 255,
        "crop": None,
        "source": 0,
        # ~~~~~~~~~ Processing Settings ~~~~~~~~~#
        "num_frames": 10,
        # ~~~~~~~~~ Run specific comments ~~~~~~~~~#
        "comments": "just testing",  # something so you can identify runs later, if you wish
    }

    os.makedirs(os.path.join(meta["home_dir"], os.path.dirname(meta["meta_file"])))
    with open(os.path.join(meta["home_dir"], meta["meta_file"]), "w") as f:
        json.dump(meta, f)

    return meta


np.set_printoptions(precision=2, suppress=True)


def main():
    meta = make_meta()

    with common.make_robot() as robot, common.make_sensor(meta) as sensor:
        # print("now clearing")
        # results = robot.sync_robot.clearAlarms()

        print("initing robot")
        common.init_robot(robot, meta, do_homing=False)

        print("Main code...")

        # _, tap_1 = ex.processed_tap_at([-5,0], 0, meta)
        tap_1 = common.tap_at([-5, 0], 0, robot, sensor, meta)

        print("collected:")
        print(tap_1)

        common.go_home(robot, meta)

        # save data
        common.save_data(tap_1, meta, "tap_1.json")

        print(np.shape(tap_1))
        # show plot of taps taken (note, this pauses program until graph is closed)
        for frame in tap_1:
            plt.scatter(frame[:, 0], frame[:, 1])
        plt.axis("equal")
        plt.show()

        # _, tap_1 = ex.processed_tap_at([-10,0], 0, meta)
        tap_2 = common.tap_at([-10, 0], 0, robot, sensor, meta)

        print("collected:")
        print(tap_2)

        common.go_home(robot, meta)

        # save data
        common.save_data(tap_2, meta, "tap_2.json")

        print(np.shape(tap_2))
        # show plot of taps taken (note, this pauses program until graph is closed)
        for frame in tap_2:
            plt.scatter(frame[:, 0], frame[:, 1])
        plt.axis("equal")
        plt.show()


    print("Done, exiting")



if __name__ == "__main__":
    main()
