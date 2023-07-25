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

# this maybe to to be "core" instead of "tactip_toolkit_dobot" for some setups
import tactip_toolkit_dobot.experiments.min_example.common as common


np.set_printoptions(precision=2, suppress=True)


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
        "home_pose": [170, 0, -150+53, 0, 0, 0],  # choose a safe "resting" pose (in base frame)
        "work_frame": [173, -5, -189 +53, 0, 0, 0],  # experiment specific start point (in base frame)
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


def main():
    # make the dictionary of metadata about current experiment and save it to
    # a file ("meta.json"). "meta" stores important variables specific to the current
    # experiment, allowing a record of what settings were used in any experiment
    # run. Variables in meta should be set manually and not changed through code.
    # Important (i.e. most) variables should be set in meta so they can be saved
    # to file and referenced in code - the main exception is data collected throughout
    # an experiment, which should be saved elsewhere (i.e. using common.save_data() ).
    # NOTE, it is important to check the variables in "make_meta()" are correct for
    # your setup (sensor settings, robot frames, "home_dir" paths etc)
    meta = make_meta()

    # "with" statement is important so that the __exit__() function is called on
    # leaving the "with" block. In this case, leaving the "with" block causes
    # the robot and sensor to disconnect (this happens even when an error causes
    # early termination of code)
    with common.make_robot() as robot, common.make_sensor(meta) as sensor:

        # set velocities and frames of robot, and make it go to safe home position
        # as defined in meta. Optionally, can perform homing which calibrates
        # robot joint rotations
        common.init_robot(robot, meta, do_homing=True)

        print("Main code...")

        # # Move to origin of work frame
        # robot.move_linear((0, 0, 0, 0, 0, 0))

        # # do a twist
        # robot.move_linear((0, 0, 0, 0, 0, 50))
        # robot.move_linear((0, 0, 0, 0, 0, -50))
        # robot.move_linear((0, 0, 0, 0, 0, 0))

        # do a tap at [x,y] = [0,0], relative to work frame, with a sensor
        # rotation of 0 using the motion defined in meta (as "tap_move"),
        # returning pin locations as "keypoints"
        keypoints = common.tap_at([0, 0], 0, robot, sensor, meta)

        # save the data to file (default file name is "data.json")
        common.save_data(keypoints, meta)

        # show plot of tap taken (note, this pauses program until graph is closed)
        for frame in keypoints:
            plt.scatter(frame[:, 0], frame[:, 1])
        plt.axis("equal")
        plt.show()

        # return to safe home position
        common.go_home(robot, meta)

    print("Done, exiting")


if __name__ == "__main__":
    main()
