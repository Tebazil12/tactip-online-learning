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

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm


np.set_printoptions(precision=2, suppress=True)


class Experiment:
    # keeps a record of the current orientation of the sensor on the robot
    # current_rotation = None

    # actual global location in 2d (x,y) , only known with arm (so far)
    edge_locations = None

    line_locations = []

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

    def collect_line(self, new_location, new_orient, meta):
        """
        Collect a straight line of taps at 90deg to new_orient, at
        displacements specified in meta.

        :param new_orient: needs to be in RADIANS!
        :returns: the best frames from each tap on the line (as n_taps x (2xn_pins) x 1)
        """
        #see if this fixes first tap being wierd...
        # self.processed_tap_at([-10,0],0,meta) # it does! why is the camera not working proplerly

        new_keypoints = [None] * len(meta["line_range"])
        best_frames = [None] * len(meta["line_range"])
        next_test_location = [None] * len(meta["line_range"])

        for i, displacements in enumerate(meta["line_range"]):

            next_test_location[i] = new_location + displacements * np.array(
                [np.cos(new_orient), np.sin(new_orient)]
            )

            best_frames[i], new_keypoints[i] = self.processed_tap_at(
                next_test_location[i], new_orient, meta
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
        common.save_data(
            next_test_location, meta, name="location_line_" + n_saves_str + ".json"
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
        position_rounded = [round(num, 1) for num in position]
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
        full_path = os.path.join(meta["home_dir"], part_path, "dissim_prof_corrected.png")
        plt.savefig(full_path)


        # use orientation and location to find real location in 2d space
        edge_location = location + offset * np.array(
                [np.cos(orient), np.sin(orient)]
            )

        corrected_disps = np.reshape(corrected_disps, (np.shape(corrected_disps)[0], 1))

        return edge_location, corrected_disps

    def processed_tap_at(
        self, new_location, new_orient, meta, neutral_tap=True, selection_criteria="Max"
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
            new_location, new_orient, self.robot, self.sensor, meta
        )

        self.add_to_alldata(
            keypoints,
            [new_location[0], new_location[1], np.rad2deg(new_orient)],
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
        self.neutral_tap, _ = self.processed_tap_at(
            [-20, -80], 0, meta, selection_criteria="Mean", neutral_tap=None
        )
        # tODO, rework taps so can set z # TODO tap motion is not needed

        common.save_data(self.neutral_tap, meta, name="neutral_tap.json")
        # return neutral_tap


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
        "num_frames": 15,
        # ~~~~~~~~~ Contour following vars ~~~~~~~~~#
        "robot_type": "arm",  # or "quad"
        "MAX_STEPS": 3,
        "STEP_LENGTH": 5, # nb, opposite direction to matlab experiments
        "line_range": np.arange(-10, 11, 2).tolist(),  # in mm
        "collect_ref_tap": True,
        "ref_location": [0, 0, 0],  # [x,y,sensor angle]
        # ~~~~~~~~~ Run specific comments ~~~~~~~~~#
        "comments": "first run with main loop instead of tests",  # so you can identify runs later
    }

    os.makedirs(os.path.join(meta["home_dir"], os.path.dirname(meta["meta_file"])))
    with open(os.path.join(meta["home_dir"], meta["meta_file"]), "w") as f:
        json.dump(meta, f)

    return meta


def explore(robot_type, ex):
    pass


def find_first_orient():
    # do 3 taps to find new_orient

    # find best frames
    # set new_location too
    return 0, [0, 0]  # TODO implement real!


def next_sensor_placement(ex, meta):
    """ New_orient needs to be in radians. """

    if ex.edge_locations is None:
        new_orient, new_location = find_first_orient()  # TODO
    else:
        if len(ex.edge_locations) == 1:
            # use previous angle
            new_orient = ex.current_rotation  # taken from current robot pose
        else:
            # interpolate previous two edge locations to find orientation
            step = np.array(ex.edge_locations[-1]) - np.array(ex.edge_locations[-2])
            new_orient = -np.arctan2(step[0], step[1])

        step_vector = meta["STEP_LENGTH"] * np.array(
            [np.sin(-new_orient), np.cos(-new_orient)]
        )
        new_location = ex.edge_locations[-1] + step_vector

    return new_orient, new_location


def plot_all_movements(ex):
    all_tap_positions_np = np.array(ex.all_tap_positions)
    pos_xs = all_tap_positions_np[:, 0]
    pos_ys = all_tap_positions_np[:, 1]
    n = range(len(pos_xs))
    plt.plot(pos_xs, pos_ys, "k")
    plt.scatter(pos_xs, pos_ys, color="k")
    ax = plt.gca()
    [ax.annotate(x[0], (x[1], x[2])) for x in np.array([n, pos_xs, pos_ys]).T]

    for line in ex.line_locations:
        line_locations_np = np.array(line)
        plt.plot(line_locations_np[:, 0], line_locations_np[:, 1], "g")
        plt.scatter(line_locations_np[:, 0], line_locations_np[:, 1], color="g")

    plt.show()


def main():
    meta = make_meta()
    ex = Experiment()

    np.set_printoptions(precision=2, suppress=True)

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
        model = None #init when first line of data collected
        n_lines_in_model = 0

        for current_step in range(0, meta["MAX_STEPS"]):
            print("------------ Main Loop -----------------")

            new_orient, new_location = next_sensor_placement(
                ex, meta
            )  # todo: make sure implemented

            if collect_more_data is False:  # should only skip on first loop
                # do single tap
                tap_1, _ = ex.processed_tap_at(new_location, new_orient, meta)

                # todo: predict distance to edge

                # todo: if exceed sensible limit
                # todo: ### collect_more_data = True

                if collect_more_data is False:

                    # todo: move distance
                    # todo: predict again

                    # todo: if bad
                    # todo: ### collect_more_data = True
                    # todo: else
                    # todo: ### note which to add location to list
                    edge_location = [0, 10*current_step]  # todo, add real logic

            if collect_more_data is True:
                new_taps = ex.collect_line(new_location, new_orient, meta)
                edge_location, adjusted_disps = ex.find_edge_in_line(
                    new_taps, ref_tap, new_location, new_orient, meta
                )

                if model is None:
                    model = gplvm.GPLVM(adjusted_disps, np.array(new_taps))
                    print(f"model inited with ls: {str(model.ls)} sigma_f: {str(model.sigma_f)}")
                    print(f"model data shape: x={np.shape(model.x)}, y={np.shape(model.y)}")
                    print(model.__dict__)

                    n_lines_str = str(n_lines_in_model).rjust(3, "0")
                    common.save_data(
                        model.__dict__, meta, name="gplvm_" + n_lines_str + ".json"
                    )
                    n_lines_in_model = n_lines_in_model +1

                # todo add new data with adjusted labels to gplvm model

            # actually add location to list (so as to not repeat self)
            if ex.edge_locations is None:
                ex.edge_locations = []
            print("edge location" + str(edge_location))
            ex.edge_locations.append(edge_location)

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

            collect_more_data = False  # last thing in loop, reset for next loop

        common.go_home(ex.robot, meta)

    # save the final set of data
    common.save_data(ex.all_raw_data, meta, name="all_data_final.json")
    common.save_data(ex.all_tap_positions, meta, name="all_positions_final.json")

    # plot results
    plot_all_movements(ex)

    print("Done, exiting")


if __name__ == "__main__":
    main()
