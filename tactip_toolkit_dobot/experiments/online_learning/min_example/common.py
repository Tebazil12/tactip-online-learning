"""
Author: Elizabeth A. Stone (lizzie.stone@brl.ac.uk)

Use:
Compilation of commonly used functions in tactile experiments. These are
generic so as to be easily used in many different experiments. See just_tap.py
for an example of how these functions are used.

"""

import os
import json
import time

import numpy as np
import copy

from tactip_toolkit_dobot.core.sensor.preproc_video_camera import CvPreprocVideoCamera

from cri.robot import AsyncRobot
from cri_dobot.robot import SyncDobot
from cri_dobot.controller import dobotMagicianController
# from cri.robot import SyncRobot
# from cri.dobot.mg400_client import Mg400Client
# from cri.controller import Mg400Controller

from vsp.video_stream import CvImageOutputFileSeq, CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.tracker import NearestNeighbourTracker
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView
from vsp.processor import (
    CameraStreamProcessor,
    AsyncProcessor,
)


def make_robot():
    """ Create and return robot instance """
    return AsyncRobot(SyncDobot(dobotMagicianController()))
    # return AsyncRobot(SyncRobot(Mg400Client()))
    # return SyncRobot(Mg400Controller(ip='172.16.0.2'))
    # return SyncRobot(Mg400Controller(ip='192.168.1.6'))
    # return SyncRobot(Mg400Controller(ip='164.11.72.19'))

    # return AsyncRobot(SyncRobot(Mg400Controller()))

def make_sensor(meta, keypoints=None):
    """
    Set up camera, setting the exposure, brightness etc and clearing the buffer
    until the settings are implemented (hacky implementation). Then create and
    return tactile sensor instance using settings from meta data.
    Current sensor is initialised to use pin detection with nearest neighbour
    tracking.
    """
    camera = CvPreprocVideoCamera(
        source=meta["source"],
        crop=meta["crop"],
        exposure=meta["exposure"],
        brightness=meta["brightness"],
        contrast=meta["contrast"],
        # api_name="V4L2"
    )  # CvVideoCamera(source=1)
    buffersize = camera.get_property("PROP_BUFFERSIZE")
    print(buffersize)
    buffersize = camera.set_property("PROP_BUFFERSIZE", 1)
    print(buffersize)
    buffersize = camera.get_property("PROP_BUFFERSIZE")
    print(buffersize)
    print(camera.camera_api)
    # asdgfa

    for _ in range(5):
        camera.read()  # Hack - camera transient
    return AsyncProcessor(
        CameraStreamProcessor(
            camera=camera,
            pipeline=[
                CvBlobDetector(
                    min_threshold=meta["min_threshold"],
                    max_threshold=meta["max_threshold"],
                    filter_by_color=meta["filter_by_color"],
                    blob_color=meta["blob_color"],
                    filter_by_area=meta["filter_by_area"],
                    min_area=meta["min_area"],
                    max_area=meta["max_area"],
                    filter_by_circularity=meta["filter_by_circularity"],
                    min_circularity=meta["min_circularity"],
                    filter_by_inertia=meta["filter_by_inertia"],
                    min_inertia_ratio=meta["min_inertia_ratio"],
                    filter_by_convexity=meta["filter_by_convexity"],
                    min_convexity=meta["min_convexity"],
                ),
                NearestNeighbourTracker(
                    threshold=meta["nntracker_threshold"], keypoints=keypoints
                ),
                KeypointEncoder(),
            ],
            view=KeypointView(color=meta["kpview_colour"]),
            display=CvVideoDisplay(name="sensor"),
            writer=CvImageOutputFileSeq(),
        )
    )


def save_data(data, meta, name=None):
    """
    Save data to a json file (named data.json) in the same directory as
    meta data file was saved. Data can be a list, np.array, dictionary (note, when adding
    np.arrays to dictionary use <array_name>.tolist() as json.dump cannot handle
    np.arrays), or anything that can be handled by json.dump
    """
    if data is not None:

        data_copy = copy.deepcopy(data) # do not want to mutate original object!

        if type(data_copy) is list:
            # horrible hack to make sure all sub lists that may be np.arrays get converted to lists
            data_copy = np.array(data_copy)
            data_copy = data_copy.tolist()

        elif type(data_copy) is np.ndarray:
            # make sure np.array becomes list
            data_copy = data_copy.tolist()

        elif (
                type(data_copy) is dict
        ):  # dictionaries are awkward, every entry must be json serialisable
            for thing in data_copy:
                if type(data_copy[thing]) is np.ndarray:
                    data_copy[thing] = data_copy[thing].tolist()

        # remove meta.json bit to add new name
        part_path, _ = os.path.split(meta["meta_file"])

        if name is not None:
            with open(os.path.join(meta["home_dir"], part_path, name), "w") as f:
                json.dump(data_copy, f)
        else:
            with open(os.path.join(meta["home_dir"], part_path, "data.json"), "w") as f:
                json.dump(data_copy, f)


def load_data(file_path):
    """
    Load the json file and return its contents as data. Note, empty files will
    cause errors, as will anything not json formatted.
    """

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def init_robot(robot, meta, do_homing=False):
    """
    Set tcp, speeds and frames using the parameters from meta data, then move
    arm to home location, then switch the robot back to the work frame.
    """

    # Set TCP, linear speed,  angular speed and coordinate frame
    robot.tcp = meta["robot_tcp"]
    robot.linear_speed = meta["linear_speed"]
    robot.angular_speed = meta["angular_speed"]

    if do_homing:
        # calibrate arm
        go_home(robot, meta)  # so not out wide and stupid
        robot.sync_robot.perform_homing()

    go_home(robot, meta)


def go_home(robot, meta):
    """ Go to safe home position """

    robot.coord_frame = meta["base_frame"]
    robot.move_linear(meta["home_pose"])
    # Return to work frame
    robot.coord_frame = meta["work_frame"]


def tap_at(location, orientation, robot, sensor, meta, height=0):
    """
    Move to location and tap with the given orientation, returning keypoints

    :param orientation: must be in DEGREES (not radians)
    """
    move_to(location, orientation,robot, meta, height)

    # change coordinate frame so origin is the new position
    robot.coord_frame = meta["base_frame"]
    robot.coord_frame = robot.pose

    print("tapping")

    # tap relative to new origin, using 2 part tap movement specified in meta
    robot.move_linear(meta["tap_move"][0])
    keypoints = sensor.process(
        meta["num_frames"] + 1,
        start_frame=1,
        outfile="/home/lizzie/git/tactip_toolkit_dobot/data/frames.png",
    )
    robot.move_linear(meta["tap_move"][1])

    # return to work frame
    robot.coord_frame = meta["work_frame"]

    # print("type keypoints")
    # print(type(keypoints))
    time.sleep(0.7)

    keypoints = np.around(keypoints, 2)  # don't really need more precision than this

    return keypoints[:, :, 0:2]  # remove weird third column of data

def move_to(location, orientation,robot, meta,height=0):
    location_1 = location[1]*0.721921#0.765#0.7846 # fix for dobot registering 10mm as 8mm in y axis
    full_cartesian_location = np.array([location[0], location_1, height, 0, 0, orientation])

    # safety checks for workspace - NB, defined relative to workframe, not
    # base frame, therefore will not be safe if workspace changes too much
    x_min = 173 + -50
    x_max = 173 + 90
    y_min = -5 + -100
    y_max = -5 + 100
    # if (
    #     meta["work_frame"][0] + location[0] < x_min
    #     or meta["work_frame"][0] + location[0] > x_max
    #     or meta["work_frame"][1] + location_1 < y_min
    #     or meta["work_frame"][1] + location_1 > y_max
    # ):
    #     raise NameError(f"Location {location[0]} {location_1} outside safe zone ({x_min}:{x_max},{y_min}:{y_max})")

    if orientation > 110 or orientation < -110:
        raise NameError(f"Orientation {orientation} of end-effector out of bounds")

    # also check its within robot limits - not sure how well this works...
    if robot.sync_robot.check_pose_is_valid(full_cartesian_location) is False:
        raise NameError("Invalid pose for robot")

    # make sure in work frame - should already be set, but just to make sure
    robot.coord_frame = meta["work_frame"]

    # move to given position
    robot.move_linear(full_cartesian_location)
