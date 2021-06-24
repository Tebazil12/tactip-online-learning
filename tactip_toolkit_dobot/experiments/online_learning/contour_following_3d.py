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
import matplotlib.ticker as ticker
from matplotlib.patches import Wedge
from mpl_toolkits.mplot3d import Axes3D

import atexit

import tactip_toolkit_dobot.experiments.min_example.common as common

# import tactip_toolkit_dobot.experiments.online_learning.experiment as experiment

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm


np.set_printoptions(precision=2, suppress=True)


class Experiment:
    # keeps a record of the current orientation of the sensor on the robot
    # current_rotation = None

    # actual global location in 2d (x,y) , only known with arm (so far)
    edge_locations = None
    edge_height = None  # should be same length as edge locations

    line_locations = []
    line_height_locations= []

    neutral_tap = None

    # all taps, in order of collection only (no nested organisation), with all
    # frames
    all_raw_data = None
    all_tap_positions = None

    robot = None
    sensor = None

    _num_line_saves = 0

    def __init__(self):
        pass

    def bootstrap(self):
        pass

    @property
    def num_line_saves(self):
        self._num_line_saves = self._num_line_saves + 1
        return self._num_line_saves

    @property
    def current_rotation(self):
        """
        :return: current rotation in DEGREES
        """
        current_pose = self.robot.pose
        return round(current_pose[5], 2)

    def displace_along_line(self, location, distance, orient):
        # location: start point in x,y
        # distance: a (scalar, xy) length to go from location
        # orient: (xy) orientation at which the displacement is applied
        # returns:
        #   new_location: the location at distance and orient from location

        return location + distance * np.array([np.cos(orient), np.sin(orient)])

    def collect_line(self, new_location, new_orient, meta, height=0):
        """
        Collect a straight line of taps at 90deg to new_orient, at
        displacements specified in meta.

        :param new_orient: needs to be in RADIANS!
        :returns: the best frames from each tap on the line (as n_taps x (2xn_pins) x 1)
        """
        # see if this fixes first tap being wierd...
        # self.processed_tap_at([-10,0],0,meta) # it does! why is the camera not working proplerly

        new_keypoints = [None] * len(meta["line_range"])
        best_frames = [None] * len(meta["line_range"])
        next_test_location = [None] * len(meta["line_range"])
        new_heights = [height] * len(meta["line_range"])

        for i, displacements in enumerate(meta["line_range"]):

            next_test_location[i] = self.displace_along_line(
                new_location, displacements, new_orient
            )
            # new_location + displacements * np.array(
            # [np.cos(new_orient), np.sin(new_orient)]
            # )

            best_frames[i], new_keypoints[i] = self.processed_tap_at(
                next_test_location[i], new_orient, meta, height=height
            )

        # save line of data
        # file_name = "data_line_" + str(self.num_line_saves).rjust(3, "0")
        n_saves_str = str(self.num_line_saves).rjust(3, "0")
        new_keypoints_np = np.array(new_keypoints)  # needed so can json.dump properly
        common.save_data(
            new_keypoints_np.tolist(), meta, name="data_line_" + n_saves_str + ".json"
        )

        self.line_locations.append(next_test_location)
        print(self.line_locations)
        self.line_height_locations.append(new_heights)
        common.save_data(
            next_test_location, meta, name="location_line_" + n_saves_str + ".json"
        )
        common.save_data(
            new_heights, meta, name="height_line_" + n_saves_str + ".json"
        )

        return best_frames

    def collect_height_line(self, new_location, new_orient, meta, start_height):
        """
        Collect a straight line of taps at heights specified in meta. (does
        not currently support sparse grids...)

        :param new_orient: needs to be in RADIANS!
        :returns: the best frames from each tap on the line (as n_taps x (2xn_pins) x 1)
        """
        # see if this fixes first tap being wierd...
        # self.processed_tap_at([-10,0],0,meta) # it does! why is the camera not working proplerly

        new_keypoints = [None] * len(meta["line_range"])
        new_heights = [None] * len(meta["line_range"])
        best_frames = [None] * len(meta["line_range"])
        next_test_location = [new_location] * len(meta["line_range"])

        for i, height_offset in enumerate(meta["height_range"]):
            new_heights[i] = start_height+height_offset

            best_frames[i], new_keypoints[i] = self.processed_tap_at(
                new_location, new_orient, meta, height=new_heights[i]
            )


        # save line of data
        # file_name = "data_line_" + str(self.num_line_saves).rjust(3, "0")
        n_saves_str = str(self.num_line_saves).rjust(3, "0")
        new_keypoints_np = np.array(new_keypoints)  # needed so can json.dump properly
        common.save_data(
            new_keypoints_np.tolist(), meta, name="data_line_" + n_saves_str + ".json"
        )

        self.line_locations.append(next_test_location)
        self.line_height_locations.append(new_heights)
        print(self.line_locations)
        common.save_data(
            next_test_location, meta, name="location_line_" + n_saves_str + ".json"
        )
        common.save_data(
            new_heights, meta, name="height_line_" + n_saves_str + ".json"
        )

        return best_frames


    def add_to_alldata(self, keypoints, position):
        """

        :param keypoints: needs to be raw data from the sensor, a single tap only
        :return: void
        """
        if self.all_raw_data is None:
            self.all_raw_data = []  # can't append to None, but needs to start as None
        self.all_raw_data.append(keypoints)

        if self.all_tap_positions is None:
            self.all_tap_positions = []
        position_rounded = [round(num, 2) for num in position]
        self.all_tap_positions.append(position_rounded)

    def find_edge_in_line(self, taps, ref_tap, location, orient, meta):
        """
        Identify the location of the edge
        :param taps: taps needs to be a line of processed taps taken in order
        at known spacial intervals (as held in meta)
        :param keypoints: raw data from a line of taps
        :param meta: the range of tap locations is held in here (assuming always the same...)
        :return:
        """

        # get dissim profile
        dissim_profile = dp.calc_dissims(
            np.array(taps), ref_tap
        )  # taps needs casting as the eulicd distance can be done on all at once (instead of looping list)

        # plot dissim profile (NB, pauses experiemnt)
        plt.plot(meta["line_range"], dissim_profile)
        # plt.show()

        # save image of graph - NB, this gets overwritten with every new line
        # remove meta.json bit to add new name
        part_path, _ = os.path.split(meta["meta_file"])
        full_path = os.path.join(meta["home_dir"], part_path, "dissim_prof.png")
        plt.savefig(full_path)

        # find min in profile
        corrected_disps, offset = dp.align_radius(
            np.array(meta["line_range"]), dissim_profile
        )

        print(offset)
        plt.plot(meta["line_range"], dissim_profile)
        plt.plot(corrected_disps, dissim_profile)
        # plt.show()
        # remove meta.json bit to add new name
        part_path, _ = os.path.split(meta["meta_file"])
        full_path = os.path.join(
            meta["home_dir"], part_path, "dissim_prof_corrected.png"
        )
        plt.savefig(full_path)
        plt.close()

        # use orientation and location to find real location in 2d space
        edge_location = location + offset * np.array([np.cos(orient), np.sin(orient)])

        corrected_disps = np.reshape(corrected_disps, (np.shape(corrected_disps)[0], 1))

        return edge_location, corrected_disps

    def find_edge_in_line_height(self, taps, ref_tap, base_height, meta, plot_on=False):
        """
        Identify the location of the edge
        :param taps: taps needs to be a line of processed taps taken in order
        at known spacial intervals (as held in meta)
        :param keypoints: raw data from a line of taps
        :param meta: the range of tap locations is held in here (assuming always the same...)
        :return:
        """

        # get dissim profile
        dissim_profile = dp.calc_dissims(
            np.array(taps), ref_tap
        )  # taps needs casting as the eulicd distance can be done on all at once (instead of looping list)

        if plot_on:
            # plot dissim profile (NB, pauses experiemnt)
            plt.plot(meta["line_range"], dissim_profile)
            # plt.show()

            # save image of graph - NB, this gets overwritten with every new line
            # remove meta.json bit to add new name
            part_path, _ = os.path.split(meta["meta_file"])
            full_path = os.path.join(meta["home_dir"], part_path, "dissim_prof.png")
            plt.savefig(full_path)

        # find min in profile
        corrected_heights, offset = dp.align_radius(
            np.array(meta["height_range"]), dissim_profile
        )

        print(offset)

        if plot_on:
            plt.plot(meta["line_range"], dissim_profile)
            plt.plot(corrected_heights, dissim_profile)
            # plt.show()
            # remove meta.json bit to add new name
            part_path, _ = os.path.split(meta["meta_file"])
            full_path = os.path.join(
                meta["home_dir"], part_path, "dissim_prof_corrected.png"
            )
            plt.savefig(full_path)
            plt.close()

        # find real height (not just relative to line)
        edge_height = base_height + offset

        corrected_heights = np.reshape(corrected_heights, (np.shape(corrected_heights)[0], 1))

        return edge_height, corrected_heights

    def processed_tap_at(
        self,
        new_location,
        new_orient,
        meta,
        neutral_tap=True,
        selection_criteria="Max",
        height=0,
    ):
        """

        :param new_location:
        :param new_orient: needs to be in RADIANS
        :param meta:
        :param neutral_tap: has 3 options - True (use self.neutral tap),
        None (use no neutral tap, just the current tap), and a np.array of same length
        as tap (use this instead of defaults)
        :return:
        """
        # TODO neutral tap should be the default, though the None option for best
        # frame should still be available
        keypoints = common.tap_at(
            new_location,
            round(np.rad2deg(new_orient), 2),
            self.robot,
            self.sensor,
            meta,
            height=height,
        )

        self.add_to_alldata(
            keypoints,
            [
                new_location[0],
                new_location[1],
                round(np.rad2deg(new_orient), 2),
                height,
            ],
        )

        if neutral_tap is True:
            neutral_tap = self.neutral_tap

        best_frames = dp.best_frame(
            keypoints, neutral_tap=neutral_tap, selection_criteria=selection_criteria
        )
        return best_frames, keypoints

    def collect_ref_tap(self, meta):
        ref_tap, _ = self.processed_tap_at(
            [meta["ref_location"][0], meta["ref_location"][1]],
            meta["ref_location"][2],
            meta,
        )
        common.save_data(ref_tap, meta, "ref_tap.json")
        return ref_tap

    def collect_neutral_tap(self, meta):
        # collect neutral, non-contact position (as reference for other taps)
        offset = meta["work_frame_offset"]

        self.neutral_tap, _ = self.processed_tap_at(
            [-20 - offset[0], -(-80) + offset[1]],
            0,
            meta,
            selection_criteria="Mean",
            neutral_tap=None,
        )
        # self.neutral_tap, _ = self.processed_tap_at(
        #     [-20 , -(-80)], 0, meta, selection_criteria="Mean", neutral_tap=None
        # )
        # tODO, rework taps so can set z # TODO tap motion is not needed

        common.save_data(self.neutral_tap, meta, name="neutral_tap.json")
        # return neutral_tap

    def save_final_data(self):
        # save the final set of data
        if state.model is not None:  # crude way of telling if things are inited
            common.save_data(self.all_raw_data, state.meta, name="all_data_final.json")
            common.save_data(
                self.all_tap_positions, state.meta, name="all_positions_final.json"
            )
            common.save_data(
                self.edge_locations, state.meta, name="all_edge_locs_final.json"
            )
            common.save_data(
                self.edge_height, state.meta, name="all_edge_heights_final.json"
            )

            common.save_data(state.model.__dict__, state.meta, name="gplvm_final.json")

    def make_graphs_final(self):
        if state.model is not None:  # crude way of telling if things are inited
            # plot results
            plot_all_movements(self, state.meta)

    def make_gplvm_graph_final(self):
        if state.model is not None:  # crude way of telling if things are inited
            # plot results
            plot_gplvm(state.model, state.meta)


def make_meta(file_name=None, stimuli_name=None, extra_dict=None):
    """
    Make dictionary of all meta data about the current experiment, and
    save to json file.
    """
    if file_name is None:
        meta_file = os.path.join(
            "online_learning",
            os.path.basename(__file__)[:-3]
            + "_"
            + time.strftime("%Yy-%mm-%dd_%Hh%Mm%Ss"),
            "meta.json",
        )
    else:
        meta_file = os.path.join(
            "online_learning",
            os.path.basename(file_name)[:-3]
            + "_"
            + time.strftime("%yy-%mm-%dd_%Hh%Mm%Ss"),
            "meta.json",
        )
    data_dir = os.path.dirname(meta_file)

    if stimuli_name is None:  # so can be overwritten in test files
        stimuli_name = "flower"
        # stimuli_name ="70mm-circle"

    if stimuli_name == "70mm-circle":
        stimuli_height = -190
        x_y_offset = [35, -35]
        # x_y_offset = [0, 0]
        max_steps = 20

    elif stimuli_name == "105mm-circle":
        stimuli_height = -190 + 2
        # x_y_offset = [57.5, -57.5]
        x_y_offset = [-17.5, 0]
        max_steps = 25

    elif stimuli_name == "flower":
        stimuli_height = -190 + 2
        x_y_offset = [35, -35 - 10 - 10]
        max_steps = 15  # 30
    else:
        raise NameError(f"Stimuli name {stimuli_name} not recognised")
    # max_steps = 3 # for testing
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
        "stimuli_name": stimuli_name,
        "stimuli_height": stimuli_height,  # location of stimuli relative to base frame
        "work_frame": [
            173 + x_y_offset[0],
            -5 + x_y_offset[1],
            stimuli_height + 1,
            0,
            0,
            0,
        ],  # experiment specific start point
        "work_frame_offset": x_y_offset,
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
        "filter_by_inertia": False,
        "min_inertia_ratio": 0.5,  # 0.22,
        "filter_by_convexity": True,
        "min_convexity": 0.61,
        "nntracker_threshold": 40,
        "kpview_colour": (0, 255, 0),
        # ~~~~~~~~~ Camera Settings ~~~~~~~~~#
        "exposure": None,  # tacFoot camera doesn't support exposure
        "brightness": 255,
        "contrast": 255,
        "crop": None,
        "source": 0,
        # ~~~~~~~~~ Processing Settings ~~~~~~~~~#
        "num_frames": 1,
        # ~~~~~~~~~ Contour following vars ~~~~~~~~~#
        "robot_type": "arm",  # or "quad"
        "MAX_STEPS": max_steps,
        "STEP_LENGTH": 10,  # 5,  # nb, opposite direction to matlab experiments
        "line_range": np.arange(-10, 11, 4).tolist(),  # in mm
        "height_range": np.array(np.arange(-1, 1.0001, 0.5)).tolist(),  # in mm
        "collect_ref_tap": True,
        "ref_location": [0, 0, np.pi / 2],  # [x,y,sensor angle in rads]
        "tol": 2,  # tolerance for displacement of second tap (0+_tol)
        "tol_height": 1,  # tolerance for height of second tap (0+_tol)
        # ~~~~~~~~~ Run specific comments ~~~~~~~~~#
        "comments": "3d trials",  # so you can identify runs later
    }

    if extra_dict is not None:
        meta.update(extra_dict)

    os.makedirs(os.path.join(meta["home_dir"], os.path.dirname(meta["meta_file"])))
    with open(os.path.join(meta["home_dir"], meta["meta_file"]), "w") as f:
        json.dump(meta, f)

    part_path, _ = os.path.split(meta["meta_file"])
    full_path = os.path.join(meta["home_dir"], part_path, "post_processing/")
    os.makedirs(full_path)

    return meta


def explore(robot_type, ex):
    pass


def find_first_orient():
    # do 3 taps to find new_orient

    # find best frames
    # set new_location too
    return np.pi / 2, [0, 0], 0  # TODO implement real!
    # return 0, [0, 0]  # TODO implement real!


def next_sensor_placement(ex, meta):
    """ New_orient needs to be in radians. """

    if ex.edge_locations is None:
        new_orient, new_location, new_height = find_first_orient()  # TODO
    else:
        if len(ex.edge_locations) == 1:
            # use previous angle
            new_orient = np.deg2rad(
                ex.current_rotation
            )  # taken from current robot pose

            # use previous height
            new_height = ex.edge_height[0]

        else:
            # interpolate previous two edge locations to find orientation
            step = np.array(ex.edge_locations[-1]) - np.array(ex.edge_locations[-2])
            new_orient = -np.arctan2(step[0], step[1])

            # interpolate height
            height_step = ex.edge_height[-1] - ex.edge_height[-2]
            new_height = ex.edge_height[-1] + height_step

        step_vector = meta["STEP_LENGTH"] * np.array(
            [np.sin(-new_orient), np.cos(-new_orient)]
        )
        new_location = ex.edge_locations[-1] + step_vector

    return new_orient, new_location, new_height


def plot_all_movements(ex, meta):
    line_width = 0.5
    marker_size = 1
    ax = plt.gca()
    if meta["stimuli_name"] == "70mm-circle":
        # print small circle location
        radius = 35
        x_offset = 35 - 35
        y_offset = 0 + 35
        # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
        ang = np.linspace(np.pi / 2, -np.pi / 2, 100)
        x = x_offset + radius * -np.cos(ang)
        y = y_offset + radius * np.sin(ang)
        plt.plot(x, y, "tab:brown", linewidth=line_width)
        # y=y*.8
        # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')

        # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
        w2 = Wedge((x_offset, y_offset), radius, 90, -90, fc="tab:brown", alpha=0.5)
        ax.add_artist(w2)
    elif meta["stimuli_name"] == "flower":
        img = plt.imread("/home/lizzie/Pictures/stimulus-flower.png")
        img_cropped = img[:, 0 : int(img.shape[0] / 2), :]
        f_size = 126
        f_y_offset = -5.2
        ax.imshow(
            img_cropped,
            extent=[-f_size / 2, 0, 0 + f_y_offset, f_size + f_y_offset],
            alpha=0.5,
        )

    # print all tap locations
    all_tap_positions_np = np.array(ex.all_tap_positions)
    pos_xs = all_tap_positions_np[1:, 0]
    pos_ys = all_tap_positions_np[1:, 1]
    # pos_ys = pos_ys/0.8
    n = range(len(pos_xs))
    plt.plot(
        pos_xs, pos_ys, "k", marker="o", markersize=marker_size, linewidth=line_width
    )
    # plt.scatter(pos_xs, pos_ys, color="k", s=marker_size)

    [
        ax.annotate(
            int(x[0]), (x[1], x[2]), fontsize=1, ha="center", va="center", color="grey"
        )
        for x in np.array([n, pos_xs, pos_ys]).T
    ]

    # print data collection lines
    for line in ex.line_locations:
        line_locations_np = np.array(line)
        plt.plot(
            line_locations_np[:, 0],
            line_locations_np[:, 1],
            "r",
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
        )
        # plt.scatter(line_locations_np[:, 0], line_locations_np[:, 1], color="g",s=marker_size)

    if ex.edge_locations is not None:
        # print predicted edge locations
        all_edge_np = np.array(ex.edge_locations)
        pos_xs = all_edge_np[:, 0]
        pos_ys = all_edge_np[:, 1]
        # pos_ys = pos_ys/0.8
        n = range(len(pos_xs))
        plt.plot(
            pos_xs,
            pos_ys,
            color="#15b01a",
            marker="+",
            markersize=marker_size + 1,
            linewidth=line_width,
        )
    # plt.scatter(pos_xs, pos_ys, color="r",marker='+',s=marker_size)
    plt.gca().set_aspect("equal")

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=5)

    # axis labels
    plt.xlabel("x displacement (mm)", fontsize=5, va="top")
    plt.ylabel("y displacement (mm)", fontsize=5, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = parse_exp_name(exp_name[1])

    plt.gcf().text(
        0.01, 1.01, meta["stimuli_name"], transform=ax.transAxes, fontsize=4, alpha=0.2
    )
    plt.gcf().text(
        1,
        1.01,
        readable_name,
        transform=ax.transAxes,
        fontsize=4,
        alpha=0.2,
        ha="right",
    )
    #     # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    # ax.set(auto=True)
    xmin, xmax, ymin, ymax = plt.axis()
    print(xmax)
    plt.axis([xmin, xmax + 2, ymin, ymax])

    #
    # # Turn on the minor TICKS, which are required for the minor GRID
    # ax.minorticks_on()
    #
    # # Customize the major grid
    # ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # # Customize the minor grid
    # ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey')
    #
    # # Turn off the display of all ticks.
    # ax.tick_params(which='both', # Options for both major and minor ticks
    #                 top='off', # turn off top ticks
    #                 left='off', # turn off left ticks
    #                 right='off',  # turn off right ticks
    #                 bottom='off') # turn off bottom ticks

    # save graphs automatically
    part_path, _ = os.path.split(meta["meta_file"])
    full_path_png = os.path.join(meta["home_dir"], part_path, "all_movements_final.png")
    full_path_svg = os.path.join(meta["home_dir"], part_path, "all_movements_final.svg")
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()
    plt.clf()


def plot_gplvm(model, meta):
    # dissim vs mu vs disp
    # per line (use meta range for counting)
    ax = plt.gca(projection="3d")

    # load ref tap so can calc dissim
    part_path, _ = os.path.split(meta["meta_file"])
    full_path_png = os.path.join(meta["home_dir"], part_path, "ref_tap.json")
    ref_tap = common.load_data(full_path_png)
    ref_tap = np.array(ref_tap)
    # calc dissims for all ys in model

    if type(model.y) is list:
        print(f"wtf, model.y is a list: {model.y}")
        model.y = np.array(model.y)
    elif type(model.y) is np.ndarray:
        print(f"model.y is an array, should be fine")
    else:
        print(f"wellp, model.y is {model.y} of type {type(model.y)}")

    dissims = dp.calc_dissims(model.y, ref_tap)

    len_line = len(meta["line_range"])

    print(f"line length= {len_line}")

    for i in range(int(len(dissims) / len_line)):

        plt.plot(
            model.x[len_line * (i) : len_line * (i + 1), 0],
            model.x[len_line * (i) : len_line * (i + 1), 1],
            zs=dissims[len_line * (i) : len_line * (i + 1)],
        )
        print(i)
        ax.text(
            model.x[len_line * (i), 0],
            model.x[len_line * (i), 1],
            dissims[len_line * (i)],
            str(i),
            # fontsize=1,
            ha="center",
            va="center",
            color="grey",
        )

    # axis labels
    ax.set_xlabel("Estimated Displacement (mm)", fontsize=5, va="top")
    ax.set_ylabel("Dissimilarity", fontsize=5, va="top")
    ax.set_zlabel(r"Optimised $\phi$", fontsize=5, va="top")

    # Show the major grid lines with dark grey lines
    ax.grid(b=True, which="major", linestyle=":", color="black")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.zaxis.set_gridline_color('black')
    # Show the minor grid lines with very faint and almost transparent grey lines
    # ax.minorticks_on()
    # ax.grid(b=True, which="minor", color="#999999", linestyle="-")
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    # ax.zaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=5)

    # save graphs automatically
    # part_path, _ = os.path.split(meta["meta_file"])
    full_path_png = os.path.join(meta["home_dir"], part_path, "gplvm_final.png")
    full_path_svg = os.path.join(meta["home_dir"], part_path, "gplvm_final.svg")
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()
    plt.clf()


def parse_exp_name(name):
    # contour_following_2d_01m-22d_14h58m05s
    split_name = name.split("_")

    # time parsing
    split_name[4] = split_name[4].replace("h", ":")
    split_name[4] = split_name[4].replace("m", ":")
    split_name[4] = split_name[4].replace("s", "")

    # date parsing
    split_name[3] = split_name[3].replace("-", "/")
    split_name[3] = split_name[3].replace("m", "")
    split_name[3] = split_name[3].replace("d", "")
    split_name[3] = split_name[3].replace("y", "")

    return (
        split_name[2].upper()
        + " "
        + split_name[0].capitalize()
        + " "
        + split_name[1].capitalize()
        + " on "
        + split_name[3]
        + " at "
        + split_name[4]
    )


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


def main(ex, model, meta):

    # np.set_printoptions(precision=2, suppress=True)

    with common.make_robot() as ex.robot, common.make_sensor(meta) as ex.sensor:
        common.init_robot(ex.robot, meta, do_homing=False)

        ex.collect_neutral_tap(meta)

        # Collect / load reference tap
        if meta["collect_ref_tap"] is True:
            ref_tap = ex.collect_ref_tap(meta)
        else:
            pass
            # ref_tap = common.load_data( <some-path> )
            # todo load a ref tap, using a path specified in meta

        collect_more_data = True  # first loop should always collect data

        n_lines_in_model = 0

        for current_step in range(0, meta["MAX_STEPS"]):
            print(f"------------ Main Loop {current_step}-----------------")

            new_orient, new_location, new_height = next_sensor_placement(ex, meta)
            if collect_more_data is False:  # should only skip on first loop
                # do single tap
                tap_1, _ = ex.processed_tap_at(
                    new_location, new_orient, meta, height=new_height
                )

                # predict distance to edge
                disp_tap_1, mu_tap_1, height_tap_1 = model.optim_single_mu_disp_height(
                    tap_1
                )
                print(
                    f"tap 1 optimised as disp={disp_tap_1} and mu={mu_tap_1} and height={height_tap_1}"
                )

                # if exceed sensible limit #TODO find vals from matlab
                if (
                    -15 > disp_tap_1
                    or disp_tap_1 > 15
                    or height_tap_1 > 2
                    or height_tap_1 < -2
                ):  # todo move to meta!!!
                    print(
                        f"distance to move from tap_1 prediction (={disp_tap_1}) or height (={height_tap_1}) is outside safe range"
                    )
                    collect_more_data = True

                if collect_more_data is False:

                    # move predicted distance
                    tap_2_location = ex.displace_along_line(
                        new_location, -disp_tap_1, new_orient
                    )

                    tap_2_height = (-height_tap_1)  # todo 3d double check this logic when awake

                    tap_2, _ = ex.processed_tap_at(
                        tap_2_location, new_orient, meta, height=tap_2_height
                    )

                    # predict again
                    (
                        disp_tap_2,
                        mu_tap_2,
                        height_tap_2,
                    ) = model.optim_single_mu_disp_height(tap_2)
                    print(
                        f"tap 2 optimised as disp={disp_tap_2} and mu={mu_tap_2} and height={height_tap_2}"
                    )

                    # was model good? was it within 0+-tol?
                    tol_d = meta["tol"]
                    tol_h = meta["tol_height"]

                    if (
                        -tol_d > disp_tap_2
                        or disp_tap_2 > tol_d
                        or height_tap_2 > tol_h
                        or height_tap_2 < -tol_h
                    ):
                        print(
                            f"tap 2 pred (disp={disp_tap_2}, height={height_tap_2}) outside of tol"
                        )
                        collect_more_data = True
                    else:
                        # note which to add location to list
                        print(f"tap 2 within of tol")
                        edge_location = tap_2_location
                        edge_height = height_tap_2

            if collect_more_data is True:
                print("Collecting data line")

                # collect displacement line only
                new_taps = ex.collect_line(
                    new_location, new_orient, meta, height=new_height
                )
                edge_location, adjusted_disps = ex.find_edge_in_line(
                    new_taps, ref_tap, new_location, new_orient, meta
                )

                # todo (maybe) use plane class to track data better?

                # collect height profile (doing cross method)
                new_taps_height = ex.collect_height_line(
                    edge_location, new_orient, meta, new_height
                )
                # find minima in height profile
                edge_height, adjusted_heights = ex.find_edge_in_line_height(
                    new_taps_height, ref_tap, new_height, meta
                )

                # add taps together to make partial x = dips height, no mu yet
                plane_y = np.concatenate((new_taps, new_taps_height), axis=0)

                plane_disps = np.concatenate((adjusted_disps, np.zeros((len(adjusted_heights),1))), axis=0)
                plane_heights =np.concatenate((edge_height*np.ones((len(adjusted_disps),1)),adjusted_heights ), axis=0)

                plane_x_no_mu = np.concatenate((plane_disps, plane_heights), axis=1)




                if model is None:
                    print("Model is None, mu will be 1")
                    # set mus to 1 for first line only - elsewhere mu is optimised
                    plane_mus = 1 * np.ones((len(plane_disps),1))
                    plane_x = np.concatenate((plane_x_no_mu, plane_mus), axis=1)

                    # init model (sets hyperpars)`
                    state.model = gplvm.GPLVM(
                        plane_x, np.array(plane_y), start_hyperpars=[1, 10, 5, 5]
                    )
                    model = state.model

                else:
                    # optimise mu of line given old data and hyperpars
                    optm_mu = model.optim_line_mu(plane_x_no_mu, plane_y)

                    plane_x = dp.add_line_mu(plane_x_no_mu, optm_mu)
                    print(f"line x to add to model = {plane_x}")

                    # save line to model (taking care with dimensions...)
                    model.x = np.vstack((model.x, plane_x))
                    model.y = np.vstack((model.y, plane_y))

                print(
                    f"model inited with ls: {str(model.ls)} sigma_f: {str(model.sigma_f)}"
                )
                print(f"model data shape: x={np.shape(model.x)}, y={np.shape(model.y)}")
                # print(model.__dict__)

                n_lines_str = str(n_lines_in_model).rjust(3, "0")
                common.save_data(
                    model.__dict__, meta, name="gplvm_" + n_lines_str + ".json"
                )
                n_lines_in_model = n_lines_in_model + 1

            # actually add location to list (so as to not repeat self)
            if ex.edge_locations is None:
                ex.edge_locations = []
            print("edge location" + str(edge_location))
            ex.edge_locations.append(edge_location)

            if ex.edge_height is None:
                ex.edge_height = []
            print(f"edge height {edge_height}")
            ex.edge_height.append(edge_height)

            # todo: exit clause for returning to first tap location

            # save data every loop just to make sure data isn't lost
            step_n_str = str(current_step).rjust(3, "0")
            common.save_data(
                ex.all_raw_data, meta, name="all_data_" + step_n_str + ".json"
            )
            common.save_data(
                ex.all_tap_positions, meta, name="all_positions_" + step_n_str + ".json"
            )
            common.save_data(
                ex.edge_locations, meta, name="all_edge_locs_" + step_n_str + ".json"
            )
            common.save_data(
                ex.edge_height, meta, name="all_edge_heights_" + step_n_str + ".json"
            )

            collect_more_data = False  # last thing in loop, reset for next loop

        common.go_home(ex.robot, meta)

    state.success = True
    # save the final set of data
    common.save_data(ex.all_raw_data, meta, name="all_data_final.json")
    common.save_data(ex.all_tap_positions, meta, name="all_positions_final.json")
    common.save_data(ex.edge_locations, meta, name="all_edge_locs_final.json")
    common.save_data(ex.edge_height, meta, name="all_edge_heights_final.json")
    common.save_data(model.__dict__, meta, name="gplvm_final.json")

    # plot results
    plot_all_movements(ex, meta)
    # todo plot edge locations too

    print("Done, exiting")


class State:
    def __init__(self, model=None, success=False, meta=None):
        self.model = model  # init when first line of data collected
        self.success = success
        self.ex = Experiment()
        if meta is None:
            self.meta = make_meta()
        else:
            self.meta = meta


if __name__ == "__main__":

    state = State()
    atexit.register(state.ex.make_gplvm_graph_final)
    atexit.register(state.ex.make_graphs_final)
    atexit.register(save_final_status)
    atexit.register(state.ex.save_final_data)

    main(state.ex, state.model, state.meta)
