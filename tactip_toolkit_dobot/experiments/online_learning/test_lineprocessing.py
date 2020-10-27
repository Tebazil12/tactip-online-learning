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

import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
)
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp


np.set_printoptions(precision=2, suppress=True)


def main():
    meta = make_meta()
    ex = Experiment()

    with common.make_robot() as ex.robot, common.make_sensor(meta) as ex.sensor:
        # print("now clearing")
        # results = ex.robot.sync_robot.clearAlarms()

        print("initing robot")
        common.init_robot(ex.robot, meta, do_homing=False)

        print("Main code...")

        results = ex.collect_line([10, 0], np.deg2rad(-45), meta)

        print("collected")
        print(ex.current_rotation)

        common.go_home(ex.robot, meta)
        print(ex.current_rotation)


        print("collected:")
        print(results)
        #
        # print("raw data")
        # print(ex.all_raw_data)

        # print("best frames")
        # print(best_frames)

        common.save_data(ex.all_raw_data, meta, name="all_data.json")
        common.save_data(ex.all_tap_positions, meta, name="all_positions.json")



    print("Done, exiting")


if __name__ == "__main__":
    main()
