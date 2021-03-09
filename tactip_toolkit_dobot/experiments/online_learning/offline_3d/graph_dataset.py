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
    parse_exp_name
)

# from tactip_toolkit_dobot.experiments.online_learning.offline_setup.collect_dataset_3d import (
#     plot_profiles_flat,
# )

# np.set_printoptions(precision=2)#, suppress=True)

def plot_flat(dissims,meta):
    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]

    the_figure = plt.figure(figsize=(10,10))
    ax = plt.gca()

    for line_num, dissim in enumerate(dissims):
        if line_num < 19:
            angle = (line_num % num_angles)*5 -45   # todo, extract from meta

            label = str(angle)+"°"
        else:
            label = ""

        # print(label)
        # if Y Z=X/Y else Z=0
        # z= ( x / y ) if y != 0 else 0
        # print(line_num)
        if (line_num  % num_angles):
            line_colour=((line_num)  % num_angles) / num_angles
        else:
            line_colour = 0

        plt.plot(
                real_disp,
                dissims[line_num - 1],
                color=(line_colour, 0, 1 - line_colour),
                label=label
            )
    plt.legend()

    font_size = 10

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
    plt.tick_params(labelsize=font_size)

    # axis labels
    plt.xlabel("Displacement (mm)", fontsize=font_size, va="top")
    plt.ylabel("Dissimilarity", fontsize=font_size, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = parse_exp_name(exp_name[1])

    # plt.gcf().text(
    #     0.01, 1.01, meta["stimuli_name"], transform=ax.transAxes, fontsize=4, alpha=0.2
    # )
    plt.gcf().text(
        1,
        1.01,
        readable_name,
        transform=ax.transAxes,
        fontsize=font_size,
        alpha=0.2,
        ha="right",
    )
    #     # Don't allow the axis to be on top of your data
    ax.set_axisbelow(True)

    full_path_png = os.path.join(
        data_home, current_experiment, "dissim_profiles_keyed.png"
    )
    full_path_svg = os.path.join(
        data_home, current_experiment, "dissim_profiles_keyed.svg"
    )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()
    plt.clf()

def plot_seperate_heights(dissims, meta):
    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]

    the_figure = plt.figure(figsize=(20,5))

    for line_num, dissim in enumerate(dissims):
        if line_num < 19:
            angle = (line_num % num_angles)*5 -45   # todo, extract from meta

            label = str(angle)

        else:
            label = ""

        # print(label)
        # if Y Z=X/Y else Z=0
        # z= ( x / y ) if y != 0 else 0
        print(line_num)
        if (line_num  % num_angles):
            line_colour=((line_num)  % num_angles) / num_angles
        else:
            line_colour = 0

        # line_colour = ((line_num-1  % num_angles) / num_angles) if (line_num-1  % num_angles) != 0 else 0

        print(line_colour)
        if (line_num+1) is 0:
            subplot_num = 1
        else:
            subplot_num = np.ceil((line_num+1)/num_angles)

        ax =the_figure.add_subplot(1, 5, subplot_num)
        # f.set_figheight(3)
        # f.set_figwidth(3)
        # plt.subplots(1,5,figsize=(15,15))
        # plt.subplots(figsize=(5, 5))

        # line_colour = (line_num  % num_angles) / num_angles
        plt.plot(
            real_disp,
            dissim,
            color=(line_colour, 0, 1 - line_colour),
            label=label
        )
        plt.axis([-11, 11, 8, 63])

        font_size = 5
        if line_num%19 == 1:


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
            plt.tick_params(labelsize=font_size)

            # axis labels
            plt.xlabel("Displacement (mm)", fontsize=font_size, va="top")
            plt.ylabel("Dissimilarity", fontsize=font_size, va="top")

            # add identifier labels
            part_path, _ = os.path.split(meta["meta_file"])

            exp_name = part_path.split("/")
            readable_name = parse_exp_name(exp_name[1])

            # plt.gcf().text(
            #     0.01, 1.01, meta["stimuli_name"], transform=ax.transAxes, fontsize=4, alpha=0.2
            # )
            plt.gcf().text(
                1,
                1.01,
                readable_name,
                transform=ax.transAxes,
                fontsize=font_size,
                alpha=0.2,
                ha="right",
            )
            #     # Don't allow the axis to be on top of your data
            ax.set_axisbelow(True)

        if line_num <= 19:
            plt.legend(fontsize=font_size)

    full_path_png = os.path.join(
        data_home, current_experiment, "dissim_profiles_heights.png"
    )
    full_path_svg = os.path.join(
        data_home, current_experiment, "dissim_profiles_heights.svg"
    )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()
    plt.clf()


def main(ex, meta):

    neutral_tap = np.array(
        common.load_data(data_home + current_experiment + "neutral_tap.json")
    )
    ref_tap = np.array(
        common.load_data(data_home + current_experiment + "ref_tap.json")
    )

    real_disp = meta["line_range"] # nb, not copied so that reverse is persistent
    real_disp.reverse() # to match previous works


    locations = []
    lines = []
    dissims = []
    num_lines = 95
    for line_num in range(1, num_lines + 1):  # todo replace this with auto indexing files

        name_loc = "location_line_" + str(line_num).rjust(3, "0") + ".json"
        locations.append(common.load_data(data_home + current_experiment + name_loc))

        name_lines = "data_line_" + str(line_num).rjust(3, "0") + ".json"
        line = np.array(common.load_data(data_home + current_experiment + name_lines))

        # process each line to be only one frame per tap
        best_frames = []
        for tap in line:
            best_frames.append(dp.best_frame(tap, neutral_tap=neutral_tap))
        lines.append(np.array(best_frames))

        # calc dissims for each line
        dissims.append(dp.calc_dissims(lines[line_num - 1], ref_tap))

    # plot_flat(dissims,meta)
    plot_seperate_heights(dissims, meta)






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

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(state.ex, state.meta)
