# Copyright 2020 Elizabeth A. Stone
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##

# Use: n/a

import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common

# import tactip_toolkit_dobot.experiments.online_learning.experiment as experiment

# import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp


np.set_printoptions(precision=2, suppress=True)


class Experiment:
    # keeps a record of the current orientation of the sensor on the robot
    current_rotation = None

    # actual global location in 2d (x,y) , only known with arm (so far)
    edge_locations = None

    # all taps, in order of collection only (no nested organisation), with all
    # frames
    all_raw_data = None

    robot = None
    sensor = None

    def __init__(self):
        pass

    def bootstrap(self):
        pass

    def collect_line(self, new_location, new_orient, meta):
        """
        Collect a straight line of taps at 90deg to new_orient, at
        displacements specified in meta.
        """
        new_keypoints = [None] * len(meta["line_range"])
        for i, displacements in enumerate(meta["line_range"]):

            next_test_location = new_location + displacements * np.array(
                [np.cos(new_orient), np.sin(new_orient)]
            )

            new_keypoints[i] = common.tap_at(
                next_test_location, new_orient, self.robot, self.sensor, meta
            )
            self.add_to_alldata(new_keypoints[i])

        return new_keypoints

    def add_to_alldata(self, keypoints):
        """

        :param keypoints: needs to be raw data from the sensor, a single tap only
        :return: void
        """
        if self.all_raw_data is None:
            self.all_raw_data = []  # can't append to None, but needs to start as None
        self.all_raw_data.append(keypoints)

    def find_edge_in_line(self, keypoints):
        """
        Identify the location of the edge
        :param keypoints: raw data from a line of taps
        :return:
        """


        pass


def make_meta():
    """
    Make dictionary of all meta data about the current experiment, and
    save to json file.
    """

    meta_file = os.path.join(
        "online_learning",
        os.path.basename(__file__)[:-3] + "_" + time.strftime("%mm-%dd_%Hh%Mm%Ss"),
        "meta.json",
    )
    data_dir = os.path.dirname(meta_file)

    stimuli_height = -190

    meta = {
        # ~~~~~~~~~ Paths ~~~~~~~~~#
        "home_dir": os.path.join(
            "/home/lizzie/git/tactip_toolkit_dobot/data", "TacTip_dobot"
        ),  # TODO THIS WILL BREAK ON OTHER MACHINES
        "meta_file": meta_file,
        # "image_dir": None,
        # "image_df_file": None,
        # "ip": None,
        # ~~~~~~~~~ Robot movements ~~~~~~~~~#
        "robot_tcp": [0, 0, 150, 0, 0, 0],  # in mm, will change between sensors
        "base_frame": [0, 0, 0, 0, 0, 0],  # see dobot manual for location
        "home_pose": [170, 0, -150, 0, 0, 0],  # choose a safe "resting" pose
        "stimuli_height": stimuli_height,  # location of stimuli relative to base frame
        "work_frame": [
            173,
            -5,
            stimuli_height + 1,
            0,
            0,
            0,
        ],  # experiment specific start point
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
        # ~~~~~~~~~ Contour following vars ~~~~~~~~~#
        "robot_type": "arm",  # or "quad"
        "MAX_STEPS": 90,
        "STEP_LENGTH": 5,
        "line_range": np.arange(-10, 11, 5).tolist(),  # in mm
        # ~~~~~~~~~ Run specific comments ~~~~~~~~~#
        "comments": "testing",  # so you can identify runs later
    }

    os.makedirs(os.path.join(meta["home_dir"], os.path.dirname(meta["meta_file"])))
    with open(os.path.join(meta["home_dir"], meta["meta_file"]), "w") as f:
        json.dump(meta, f)

    return meta


def explore(robot_type, ex):
    pass


def find_first_orient():
    # do 3 taps to find new_orient
    # set new_location too
    pass


def next_sensor_placement(ex, meta):
    """ New_orient needs to be in radians. """

    if ex.edge_locations is None:
        new_orient, new_location = find_first_orient()
    else:
        if len(ex.edge_locations) == 1:
            # use previous angle
            new_orient = ex.current_rotation
        else:
            # interpolate previous two edge locations to find orientation
            step = ex.edge_locations[-1] - ex.edge_locations[-2]
            new_orient = -np.arctan2(step[0], step[1])

        step_vector = meta["STEP_LENGTH"] * np.array(
            [np.sin(-new_orient), np.cos(-new_orient)]
        )
        new_location = ex.edge_locations[-1] + step_vector

    return new_orient, new_location


def main():
    meta = make_meta()
    ex = Experiment()

    with common.make_robot() as ex.robot, common.make_sensor(meta) as ex.sensor:
        common.init_robot(ex.robot, meta)

        print("Main code...")

        for current_step in range(0, meta["MAX_STEPS"]):
            print("------------ Main Loop -----------------")

            # explore(meta["robot_type"], ex)

            # if meta["robot_type"] == "arm":

            new_orient, new_location = next_sensor_placement(ex, meta)

            if current_step == 0:  # first run
                new_keypoints = ex.collect_line(new_location, new_orient, meta)
                ex.find_edge_in_line(new_keypoints)
            else:
                pass
                # do single tap
                # predict distance to edge
                # if exceed sensible limit
                # collect line
                # move distance
                # predict again
                # if bad
                # collect line

            # elif meta["robot_type"] == "quad":
            #     pass

        # # Move to origin of work frame
        # robot.move_linear((0, 0, 0, 0, 0, 0))
        #
        # # # do a twist
        # # robot.move_linear((0, 0, 0, 0, 0, 50))
        # # robot.move_linear((0, 0, 0, 0, 0, -50))
        # # robot.move_linear((0, 0, 0, 0, 0, 0))
        #
        # # do a tap, recording pin positions
        # robot.move_linear(meta["tap_move"][0])
        # keypoints = sensor.process(meta["num_frames"])
        # robot.move_linear(meta["tap_move"][1])
        #
        # data = {"keypoints":keypoints.tolist()}
        # common.save_data(data, meta)
        #
        # # show plot of tap taken
        # # for i in range(0,10):
        # #     plt.scatter(keypoints[i][:,0],keypoints[i][:,1] )
        # #     plt.plot(keypoints[i][:,0],keypoints[i][:,1] )
        # # plt.axis('equal')
        # for i in range(0, len(keypoints[0])):
        #     plt.scatter(keypoints[:, i, 0], keypoints[:, i, 1])
        #     plt.plot(keypoints[:, i, 0], keypoints[:, i, 1])
        # # plt.plot(keypoints[1][:,0],keypoints[1][:,1] )
        # plt.scatter(keypoints[0][:, 0], keypoints[0][:, 1])
        # plt.axis("equal")
        # plt.show()
        #
        # common.go_home(robot, meta)
        # print("Final pose in work frame: {}".format(robot.pose))

    print("Done, exiting")


if __name__ == "__main__":
    main()
