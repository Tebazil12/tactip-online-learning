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
    x_no_mu = None # x but without mu, so mu can be optimised

    y = None  # the (processed) pin data at these points

    dissims = None

    real_height = None # baseline height in case of a plane # for use offline only
    real_angle = None # for use offline only

    def __init__(self):
        pass

    def __add__(self, other_plane):

        return_plane = Plane()
        if self.disps is not None and other_plane.disps is not None:
            return_plane.disps = np.concatenate(
                (self.disps, other_plane.disps), axis=0
            )
        else:
            return_plane.disps = None

        if self.heights is not None and other_plane.heights is not None:
            return_plane.heights = np.concatenate(
                (self.heights, other_plane.heights), axis=0
            )
        else:
            return_plane.heights = None

        if self.phis is not None and other_plane.phis is not None:
            return_plane.phis = np.concatenate(
                (self.phis, other_plane.phis), axis=0
            )
        else:
            return_plane.phis = None

        if self.x is not None and other_plane.x is not None:
            return_plane.x = np.concatenate(
                (self.x, other_plane.x), axis=0
            )
        else:
            return_plane.x = None

        if self.x_no_mu is not None and other_plane.x_no_mu is not None:
            return_plane.x_no_mu = np.concatenate(
                (self.x_no_mu, other_plane.x_no_mu), axis=0
            )
        else:
            return_plane.x_no_mu = None

        if self.y is not None and other_plane.y is not None:
            return_plane.y = np.concatenate(
                (self.y, other_plane.y), axis=0
            )
        else:
            return_plane.y = None

        if self.dissims is not None and other_plane.dissims is not None:
            return_plane.dissims = np.concatenate(
                (self.dissims, other_plane.dissims), axis=0
            )
        else:
            return_plane.dissims = None

        if self.real_height is not None and other_plane.real_height is not None:
            return_plane.real_height = np.concatenate(
                (self.real_height, other_plane.real_height), axis=0
            )
        else:
            return_plane.real_height = None

        if self.real_angle is not None and other_plane.real_angle is not None:
            return_plane.real_angle = np.concatenate(
                (self.real_angle, other_plane.real_angle), axis=0
            )
        else:
            return_plane.real_angle = None

        return return_plane

    def make_x(self):
        # if len(self.disps.shape) is not 2:
        #     raise NameError(f"disps is wrong shape to make x: is {self.disps.shape} not (*,1)")
        #
        # if len(self.heights.shape) is not 2:
        #     raise NameError(f"heights is wrong shape to make x: is {self.heights.shape} not (*,1)")
        #
        # if len(self.phis.shape) is not 1:
        #     raise NameError(f"phis is wrong shape to make x: is {self.phis.shape} not (*,)")

        # combine locs and phis (if needed, heights) - reuse other funciotn?
        self.x = np.concatenate(
            (
                self.disps,
                self.heights,
                np.array([self.phis]).T,
            ),
            axis=1,
        )
        # print(f"x is shape: {np.shape(self.x)}")

    def make_x_no_mu(self):
        self.x_no_mu = np.concatenate((self.disps, self.heights), axis=1)

    def make_all_heights(self, height):
        # make heights the same length as disp and set to given height

        if self.disps is None:
            raise NameError("disps must be defined before all_heights can be used")

        self.heights = np.array([[height] * len(self.disps)]).T

    def make_all_phis(self, phi):
        # make heights the same length as disp and set to given height

        if self.disps is None:
            raise NameError("disps must be defined before all_phis can be used")

        self.phis = [phi] * len(self.disps)

def extract_point_at(local, data, meta):
    # return the data at given height, angle, displacment
    # local=[height(mm), angle(deg),disp(mm)]
    real_disp = np.array(meta["line_range"])

    line = extract_line_at(local,data,meta)
    disp_index = np.where(real_disp == local[2])[0][0]

    the_point = Plane()
    the_point.y = line.y[disp_index]
    the_point.disps = [local[2]]
    the_point.make_all_phis(local[1])
    the_point.make_all_heights(local[0])
    the_point.make_x()
    the_point.real_height = local[0]
    the_point.real_angle = local[1]

    # print(f"the point = {the_point.__dict__}")

    return the_point

def extract_line_at(local, data, meta, dissims=None):
    # return the data at given height and angle, local=[height, angle(in deg)]

    heights = np.array(meta["height_range"])
    num_heights = len(heights)
    angles = np.array(meta["angle_range"])
    num_angles = len(angles)
    real_disp = np.array(meta["line_range"])
    num_disps = len(real_disp)

    height_index = np.where(heights == local[0])[0][0]
    # print(f"height index {height_index}")
    # print(f"local {local} local 1 {local[1]}")
    # print(f"angles = {angles}")
    angle_index = np.where(angles == local[1])[0][0]
    # print(f"angle index {angle_index}")

    index = angle_index + ((height_index) * num_angles)

    # print(f"getting index {index}")

    the_line = Plane()
    the_line.y = np.array([data[index]]).T
    the_line.disps =  np.array([real_disp]).T
    the_line.make_all_phis(local[1])
    the_line.make_all_heights(local[0])
    if dissims is not None:
        the_line.dissims = np.array([dissims[index]]).T
    # print(f"the line: {the_line.__dict__}")

    the_line.make_x()
    the_line.real_height = local[0]
    the_line.real_angle = local[1]

    return the_line


def at_plane_extract(
    local, data, meta, method="cross", cross_length=None, cross_disp=0, dissims=None
):
    local = np.array(local)
    base_line = extract_line_at(local, data, meta, dissims=dissims)



    if method == "cross":
        # take the points cross_length up and cross_length down (in height)
        # from center of line

        other_lines = []
        # other_ys = []
        height_step = 0.5  # todo, extract from meta

        for i in np.arange(
            height_step, (cross_length * height_step) + 0.000001, height_step
        ):  # todo, map to .5mm prooperly
            other_lines.append(extract_line_at(local + np.array([i, 0]), data, meta, dissims=dissims))
            # other_ys.append(other_lines[-1].y)
            other_lines.append(extract_line_at(local + np.array([-i, 0]), data, meta, dissims=dissims))
            # other_ys.append(other_lines[-1].y)

        # other_ys = np.array(other_ys)

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
    elif method == "grid":
        # take the points cross_length up and cross_length down (in height)
        # from center of line, and 4 extreme corners

        other_lines = []
        other_ys = []
        height_step = 0.5  # todo, extract from meta

        for i in np.arange(
            height_step, (cross_length * height_step) + 0.000001, height_step
        ):  # todo, map to .5mm prooperly
            # print(f"i {i}")
            other_lines.append(extract_line_at(local + np.array([i, 0]), data, meta, dissims=dissims))
            other_ys.append(other_lines[-1].y)
            other_lines.append(extract_line_at(local + np.array([-i, 0]), data, meta, dissims=dissims))
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

        # extremes of grid
        for line in other_lines[-2:]:
            base_line.y = np.concatenate(
                (base_line.y, np.array([line.y[0]])), axis=0
            )
            base_line.disps = np.concatenate(
                (base_line.disps, np.array([line.disps[0]])), axis=0
            )
            base_line.heights = np.concatenate(
                (base_line.heights, np.array([line.heights[0]])), axis=0
            )

            base_line.y = np.concatenate(
                (base_line.y, np.array([line.y[-1]])), axis=0
            )
            base_line.disps = np.concatenate(
                (base_line.disps, np.array([line.disps[-1]])), axis=0
            )
            base_line.heights = np.concatenate(
                (base_line.heights, np.array([line.heights[-1]])), axis=0
            )

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

        # print(f"baseline {base_line.__dict__}")

        return base_line

    elif method == "full":
        # full grid
        result_plane = None
        other_lines = []
        # other_ys = []
        height_step = 0.5  # todo, extract from meta

        # for i in np.arange(-(cross_length * height_step), (cross_length * height_step) + 0.000001, height_step):
        for i in meta["height_range"]:
            if result_plane is None:
                result_plane = extract_line_at(local + np.array([i, 0]), data, meta, dissims=dissims)
            else:
                # print(f"i {i}")
                other_lines.append(extract_line_at(local + np.array([i, 0]), data, meta, dissims=dissims))
            # other_ys.append(other_lines[-1].y)
            # other_lines.append(extract_line_at(local + np.array([-i, 0]), data, meta, dissims=dissims))
            # other_ys.append(other_lines[-1].y)

        # other_ys = np.array(other_ys)

        index_to_take = cross_disp
        # cross disp is index shift atm, not mm shift
        # rounds down in case of even num of taps (which really shouldn't be the case)

        # print(f"index: {index_to_take}")

        # print(
        #     f"other ys shape: {np.shape(other_ys)} slice: {np.shape(other_ys[:,index_to_take])}"
        # )
        # print(f"shape of base_line.y {np.shape(base_line.y)}")

        # extremes of grid
        # for line in other_lines[-2:]:
        #     base_line.y = np.concatenate(
        #         (base_line.y, np.array([line.y[0]])), axis=0
        #     )
        #     base_line.disps = np.concatenate(
        #         (base_line.disps, np.array([line.disps[0]])), axis=0
        #     )
        #     base_line.heights = np.concatenate(
        #         (base_line.heights, np.array([line.heights[0]])), axis=0
        #     )
        #
        #     base_line.y = np.concatenate(
        #         (base_line.y, np.array([line.y[-1]])), axis=0
        #     )
        #     base_line.disps = np.concatenate(
        #         (base_line.disps, np.array([line.disps[-1]])), axis=0
        #     )
        #     base_line.heights = np.concatenate(
        #         (base_line.heights, np.array([line.heights[-1]])), axis=0
        #     )

        for line in other_lines:
            # print("---")
            # print(f"line.y is shape {np.shape(line.y)}")
            # new_point = np.array([line.y[index_to_take]])
            # print(f"new point {np.shape(new_point)}")

            result_plane.y = np.concatenate(
                (result_plane.y, line.y), axis=0
            )
            result_plane.disps = np.concatenate(
                (result_plane.disps, line.disps), axis=0
            )
            result_plane.heights = np.concatenate(
                (result_plane.heights, line.heights), axis=0
            )
            # todo same for disp etc
            result_plane.dissims = np.concatenate(
                (result_plane.dissims, line.dissims), axis=0
            )

        result_plane.make_all_phis(None)  # could use None to show its not optimised?
        result_plane.make_x()

        # print(f"baseline = {base_line}, has vars {base_line.__dict__}")
        # print(
        #     f"baseline y is shape {np.shape(base_line.y)}, disp is {np.shape(base_line.disps)}"
        # )

        # print(f"baseline {result_plane.__dict__}")

        return result_plane


def get_calibrated_plane(local, meta, lines, optm_disps, ref_tap, num_disps):
    [adjusted_disps] = extract_line_at(local, optm_disps, meta).disps.T
    # print(f"sjusted disps= {adjusted_disps}")
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
        method="grid",
        cross_length=2,
        cross_disp=offset_index_disp,
    )

    # print(f"new taps plane {new_taps_plane.__dict__}\n of shapes y {new_taps_plane.y.shape} x {new_taps_plane.x.shape} disp {new_taps_plane.disps.shape}")

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
    # print(f"new taps plane {new_taps_plane.__dict__}\n of shapes y {new_taps_plane.y.shape} x {new_taps_plane.x.shape} disp {new_taps_plane.disps.shape}")

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


def main(ex, meta, train_or_test="train", train_folder=""):

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

        for local in [training_local_1, [0,45]]:
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
        common.save_data(model.__dict__,meta, "post_processing/"+train_folder+"gplvm_model.json")


    else: # must be testing, so load pre-trained model

        model_dict = common.load_data(path + "post_processing/"+train_folder+"gplvm_model.json")

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
            common.save_data(mus_optm,meta, "post_processing/"+train_folder+"optm_plane_mus.json")
            print(angles_of_planes)
            # mus_test = model.optim_many_mu(disp_test, y_test)

            # visulaise predictions
            plt.plot(angles_of_planes, mus_optm)
            plt.plot([-45,0,45],[0,1,2],'k:') #plot ideal relation # TODO extract from data
            plt.xlabel("real angle (degrees)")
            plt.ylabel("predicted phi")

            # save graphs automatically
            part_path, _ = os.path.split(meta["meta_file"])
            full_path_png = os.path.join(meta["home_dir"], part_path, "post_processing/"+train_folder+"phi_predictions.png")
            full_path_svg = os.path.join(meta["home_dir"], part_path, "post_processing/"+train_folder+"phi_predictions.svg")
            plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
            plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)
            plt.show()



        elif train_or_test == "test_single_taps":
            results = {
                "real":[],
                "optm":[]
            }
            for height in heights:
                for angle in angles:
                    for disp in real_disp:
                        point = extract_point_at([height,angle,disp], lines, meta)
                        disp_optm, mu_optm, height_optm = model.optim_single_mu_disp_height(point.y)
                        results["real"].append([height,angle,disp])
                        results["optm"].append([height_optm,mu_optm,disp_optm])
            print(results)
            common.save_data(results,meta, "post_processing/"+train_folder+"single_tap_results.json")

            results["real"] = np.array(results["real"])
            results["optm"] = np.array(results["optm"])

            errors = results["optm"] - results["real"]
            print(errors)


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

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_cross/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_two_cross/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_one_grid/")
    # main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_one_grid/")

    # main(state.ex, state.meta,train_or_test="train", train_folder="model_two_grid/")
    # main(state.ex, state.meta,train_or_test="test_line_angles",train_folder="model_two_grid/")
    main(state.ex, state.meta,train_or_test="test_single_taps", train_folder="model_two_grid/")
