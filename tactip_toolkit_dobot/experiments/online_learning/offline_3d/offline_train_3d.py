import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gplvm as gplvm
import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
    plot_gplvm,
    State,
    parse_exp_name,
)


class Plane:
    disps = None  # known/estimated disps to edge
    heights = None  # robot / estimated z location
    phis = None  # known / estimated sensor angle wrt edge

    x = None  # combination -> x = [locs, phis, heights]

    y = None  # the (processed) pin data at these points

    dissims = None

    real_height = None # baseline height in case of a plane # for use offline only
    real_angle = None # for use offline only

    def __init__(self):
        pass

    def make_x(self):
        # combine locs and phis (if needed, heights) - reuse other funciotn?
        self.x = np.concatenate(
            (
                np.array([self.disps]).T,
                np.array([self.heights]).T,
                np.array([self.phis]).T,
            ),
            axis=1,
        )
        # print(f"x is shape: {np.shape(self.x)}")

    def make_all_heights(self, height):
        # make heights the same length as disp and set to given height

        if self.disps is None:
            raise NameError("disps must be defined before all_heights can be used")

        self.heights = [height] * len(self.disps)

    def make_all_phis(self, phi):
        # make heights the same length as disp and set to given height

        if self.disps is None:
            raise NameError("disps must be defined before all_phis can be used")

        self.phis = [phi] * len(self.disps)


def extract_line_at(local, data, meta):
    # return the data at given height and angle, local=[height, angle(in deg)]

    heights = np.array(meta["height_range"])
    num_heights = len(heights)
    angles = np.array(meta["angle_range"])
    num_angles = len(angles)
    real_disp = np.array(meta["line_range"])
    num_disps = len(real_disp)

    height_index = np.where(heights == local[0])[0][0]
    # print(f"height index {height_index}")
    angle_index = np.where(angles == local[1])[0][0]
    # print(f"angle index {angle_index}")

    index = angle_index + ((height_index) * num_angles)

    # print(f"getting index {index}")

    the_line = Plane()
    the_line.y = data[index]
    the_line.disps = real_disp
    the_line.make_all_phis(local[1])
    the_line.make_all_heights(local[0])
    the_line.make_x()
    the_line.real_height = local[0]
    the_line.real_angle = local[1]

    return the_line


def at_plane_extract(
    local, data, meta, method="cross", cross_length=None, cross_disp=0
):
    base_line = extract_line_at(local, data, meta)

    local = np.array(local)

    if method == "cross":
        # take the points cross_length up and cross_length down (in height)
        # from center of line

        other_lines = []
        other_ys = []
        height_step = 0.5  # todo, extract from meta

        for i in np.arange(
            height_step, (cross_length * height_step) + 0.000001, height_step
        ):  # todo, map to .5mm prooperly
            other_lines.append(extract_line_at(local + np.array([i, 0]), data, meta))
            other_ys.append(other_lines[-1].y)
            other_lines.append(extract_line_at(local + np.array([-i, 0]), data, meta))
            other_ys.append(other_lines[-1].y)

        other_ys = np.array(other_ys)

        index_to_take = cross_disp
        # cross disp is index shift atm, not mm shift
        # rounds down in case of even num of taps (which really shouldn't be the case)

        # print(f"index: {index_to_take}")

        # print(
        #     f"other ys shape: {np.shape(other_ys)} slice: {np.shape(other_ys[:,index_to_take])}"
        # )
        # print(f"shape of base_line.y {np.shape(base_line.y)}")

        for line in other_lines:
            # print("---")
            # print(f"line.y is shape {np.shape(line.y)}")
            # new_point = np.array([line.y[index_to_take]])
            # print(f"new point {np.shape(new_point)}")

            base_line.y = np.concatenate(
                (base_line.y, np.array([line.y[index_to_take]])), axis=0
            )
            base_line.disps = np.concatenate(
                (base_line.disps, np.array([line.disps[index_to_take]])), axis=0
            )
            base_line.heights = np.concatenate(
                (base_line.heights, np.array([line.heights[index_to_take]])), axis=0
            )
            # todo same for disp etc

        base_line.make_all_phis(None)  # could use None to show its not optimised?
        base_line.make_x()

        # print(f"baseline = {base_line}, has vars {base_line.__dict__}")
        # print(
        #     f"baseline y is shape {np.shape(base_line.y)}, disp is {np.shape(base_line.disps)}"
        # )

        return base_line


def get_calibrated_plane(local, meta, lines, optm_disps, ref_tap, num_disps):
    adjusted_disps = extract_line_at(local, optm_disps, meta).y
    [[offset_index_disp]] = np.where(adjusted_disps == 0)
    # offset_index_disp = 4

    center_index = int(np.floor((num_disps - 1) / 2))
    if offset_index_disp != center_index:

        print(
            f"WARNING - disp offset index is {offset_index_disp}, should be {center_index}"
        )

    # print(f"lengths are {len(new_taps)} and {len(adjusted_disps)}")

    # Collect height data at disp minima (in offline, also line data)
    new_taps_plane = at_plane_extract(
        local,
        lines,
        meta,
        method="cross",
        cross_length=2,
        cross_disp=offset_index_disp,
    )

    # print(new_taps_plane.__dict__)

    # Adjust displacement of plane to edge loc #horrible hack for offline
    disp_mm_offset = new_taps_plane.disps[-1]  # should be found online in a better way!
    new_taps_plane.disps = new_taps_plane.disps - disp_mm_offset
    new_taps_plane.make_x()

    # print(new_taps_plane.__dict__)

    # Find height minima
    indexs = np.where(new_taps_plane.disps == 0)
    # print(indexs)
    # print(np.shape(new_taps_plane.y))
    height_x = new_taps_plane.x[indexs[0], :]
    height_y = new_taps_plane.y[indexs[0], :]

    # print(height_y)
    # print(height_x)

    # Adjust heights based on minima
    # calc dissims for plane
    new_taps_plane.dissims = dp.calc_dissims(new_taps_plane.y, ref_tap)
    # print(new_taps_plane.dissims)

    # seperate out just the height profile
    height_dissims = new_taps_plane.dissims[indexs[0]]
    # print(height_dissims)

    # reorder as wasn't build in correct order for profile
    height_height = height_x[:, 1]
    # print(f"height {height_height}")

    zipped = zip(height_height, height_dissims)
    # print(zipped)

    sorted_stuff = sorted(zipped)
    # print(sorted_stuff)
    sorted_stuff = np.array(sorted_stuff)

    height_dissims_sorted = sorted_stuff[:, 1]
    height_height_sorted = sorted_stuff[:, 0]

    corrected_heights, height_offset = dp.align_radius(
        height_height_sorted, height_dissims_sorted
    )

    # print(f"results={corrected_heights} offset = {height_offset}")

    # add offset to all x heights
    new_taps_plane.heights = new_taps_plane.heights - height_offset
    # print(new_taps_plane.heights)
    new_taps_plane.make_x()

    return new_taps_plane


def main(ex, meta, train_or_test="train"):

    # load data
    path = data_home + current_experiment

    neutral_tap = np.array(common.load_data(path + "neutral_tap.json"))
    ref_tap = np.array(common.load_data(path + "ref_tap.json"))

    locations = np.array(common.load_data(path + "post_processing/all_locations.json"))
    lines = np.array(common.load_data(path + "post_processing/all_lines.json"))
    dissims = np.array(common.load_data(path + "post_processing/dissims.json"))

    optm_disps = np.array(
        common.load_data(path + "post_processing/corrected_disps_basic.json")
    )

    heights = meta["height_range"]
    num_heights = len(heights)
    angles = meta["angle_range"]
    num_angles = len(angles)
    real_disp = meta["line_range"]
    num_disps = len(real_disp)

    if train_or_test == "train":
        # Find location of disp minima
        training_local_1 = [0, 0] # [height(in mm), angle(in deg)]

        for local in [training_local_1]:#, [0,45]]:
            # new_taps = extract_line_at(training_local_1, lines, meta).y

            ready_plane = get_calibrated_plane(
                local, meta, lines, optm_disps, ref_tap, num_disps
            )

            print(f"calibrated plane is: {ready_plane.__dict__}")

            if state.model is None:
                print("Model is None, mu will be 1")
                # set mus to 1 for first line only - elsewhere mu is optimised
                ready_plane.make_all_phis(1)
                ready_plane.make_x()

                print(ready_plane.__dict__)


                # init model (sets hyperpars)
                state.model = gplvm.GPLVM(ready_plane.x, ready_plane.y, start_hyperpars=[1, 10, 5,5])
                model = state.model
            else:
                optm_mu = model.optim_line_mu(ready_plane.x, ready_plane.y)

                x_line = dp.add_line_mu(ready_plane.x, optm_mu)
                print(f"line x to add to model = {x_line}")

                # save line to model (taking care with dimensions...)
                model.x = np.vstack((model.x, x_line))
                model.y = np.vstack((model.y, ready_plane.y))



        print(model.__dict__)
        common.save_data(model.__dict__,meta, "post_processing/gplvm_model.json")


    else: # must be testing, so load pre-trained model

        model_dict = common.load_data(path + "post_processing/gplvm_model.json")

        state.model = gplvm.GPLVM(
            np.array(model_dict["x"]),
            np.array(model_dict["y"]),
            sigma_f=model_dict["sigma_f"],
            ls=model_dict["ls"],
        )
        model = state.model

        print(state.model.__dict__)
        print(type(state.model))

        print("~~~ Model Loaded, Starting Testing ~~~")

        if train_or_test == "test_line_angles":
            disp_test = []
            y_test = []
            mus_optm = []
            angles_of_planes = []
            for height in [0]:#heights:
                for angle in angles:
                    ready_plane = get_calibrated_plane(
                        [height,angle], meta, lines, optm_disps, ref_tap, num_disps
                    )
                    ready_plane.make_all_phis(None) # ensure phis are none
                    ready_plane.make_x()
                    # print(f"plane={ready_plane}")
                    # print(ready_plane.__dict__)
                    # disp_test.append(ready_plane.x)
                    mu = model.optim_line_mu(ready_plane.x, ready_plane.y)
                    mus_optm.append(mu)
                    angles_of_planes.append(ready_plane.real_angle)

            print(mus_optm)
            common.save_data(mus_optm,meta, "post_processing/optm_plane_mus.json")
            print(angles_of_planes)
            # mus_test = model.optim_many_mu(disp_test, y_test)

            # visulaise predictions
            plt.plot(angles_of_planes, mus_optm)
            plt.plot([-45,0,45],[0,1,2],'k:') #plot ideal relation # TODO extract from data
            plt.xlabel("real angle (degrees)")
            plt.ylabel("predicted phi")

            # save graphs automatically
            part_path, _ = os.path.split(meta["meta_file"])
            full_path_png = os.path.join(meta["home_dir"], part_path, "post_processing/phi_predictions.png")
            full_path_svg = os.path.join(meta["home_dir"], part_path, "post_processing/phi_predictions.svg")
            plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
            plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)
            plt.show()



        elif train_or_test == "test_single_taps":
            pass
    # best_frames = dp.best_frame()

    # print(model["ls"])
    # state.model = gplvm.GPLVM(
    #     np.array(model["x"]),
    #     np.array(model["y"]),
    #     sigma_f=model["sigma_f"],
    #     ls=model["ls"],
    # )
    # print(state.model.x)
    # #
    # #
    # plot_gplvm(state.model, meta)


if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(f"Dataset: {current_experiment} using {state.meta['stimuli_name']}")

    # reverse real displacements so when main is run twice, the change is not reverted
    real_disp = state.meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works (-ve on obj, +ve free)

    state.ex = Experiment()

    main(state.ex, state.meta,train_or_test="train")
    main(state.ex, state.meta,train_or_test="test_line_angles")
