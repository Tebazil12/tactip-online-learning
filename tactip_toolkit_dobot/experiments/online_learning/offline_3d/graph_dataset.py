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

# from tactip_toolkit_dobot.experiments.online_learning.offline_setup.collect_dataset_3d import (
#     plot_profiles_flat,
# )

# np.set_printoptions(precision=2)#, suppress=True)


def plot_flat(dissims, meta, data_home=None, current_experiment=None):
    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]

    the_figure = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    for line_num, dissim in enumerate(dissims):
        if line_num < 19:
            angle = (line_num % num_angles) * 5 - 45  # todo, extract from meta

            label = str(angle) + "°"
        else:
            label = ""

        # print(label)
        # if Y Z=X/Y else Z=0
        # z= ( x / y ) if y != 0 else 0
        # print(line_num)
        if line_num % num_angles:
            line_colour = ((line_num) % num_angles) / num_angles
        else:
            line_colour = 0

        plt.plot(
            real_disp,
            dissims[line_num - 1],
            color=(line_colour, 0, 1 - line_colour),
            label=label,
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


def plot_seperate_heights(dissims, meta, data_home=None, current_experiment=None):
    num_heights = len(meta["height_range"])
    heights = meta["height_range"]
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]

    the_figure = plt.figure(figsize=(20, 5))

    for line_num, dissim in enumerate(dissims):
        if line_num < 19:
            angle = (line_num % num_angles) * 5 - 45  # todo, extract from meta

            label = str(angle) + "°"

        else:
            label = ""

        # print(label)
        # if Y Z=X/Y else Z=0
        # z= ( x / y ) if y != 0 else 0
        print(line_num)
        if line_num % num_angles:
            line_colour = ((line_num) % num_angles) / num_angles
        else:
            line_colour = 0

        # line_colour = ((line_num-1  % num_angles) / num_angles) if (line_num-1  % num_angles) != 0 else 0

        print(line_colour)
        if (line_num + 1) is 0:
            subplot_num = 1
        else:
            subplot_num = int(np.ceil((line_num + 1) / num_angles))

        ax = the_figure.add_subplot(1, 5, subplot_num)
        # f.set_figheight(3)
        # f.set_figwidth(3)
        # plt.subplots(1,5,figsize=(15,15))
        # plt.subplots(figsize=(5, 5))

        # line_colour = (line_num  % num_angles) / num_angles
        plt.plot(
            real_disp, dissim, color=(line_colour, 0, 1 - line_colour), label=label
        )
        plt.axis([-11, 11, 0, 70])

        font_size = 5
        if line_num % 19 == 1:

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

            # height = (subplot_num*0.5)-1.5
            # print(subplot_num)
            # print(1)
            # print(subplot_num-1)
            height = heights[subplot_num - 1]
            plt.title(
                f"Profiles at tap depth {height} mm from reference",
                fontsize=(font_size + 1),
            )

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


def plot_minimas(
    dissims, meta, gp_extrap=True, data_home=None, current_experiment=None, show_figs=True
):

    num_heights = len(meta["height_range"])
    heights = meta["height_range"]
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]

    the_figure = plt.figure(figsize=(20, 5))

    font_size = 5
    offsets = []
    corrected_disps = []

    # print(f"dissims len {len(dissims)} ")

    for line_num, dissim in enumerate(dissims):
        angle = (line_num % num_angles) * 5 - 45  # todo, extract from meta
        if line_num < num_angles:

            label = str(angle) + "°"

        else:
            label = ""

        # print(label)
        # if Y Z=X/Y else Z=0
        # z= ( x / y ) if y != 0 else 0
        # print(line_num)
        if line_num % num_angles:
            line_colour = ((line_num) % num_angles) / num_angles
        else:
            line_colour = 0

        # line_colour = ((line_num-1  % num_angles) / num_angles) if (line_num-1  % num_angles) != 0 else 0

        # print(line_colour)
        if (line_num + 1) is 0:
            subplot_num = 1
        else:
            subplot_num = int(np.ceil((line_num + 1) / num_angles))

        corrected_disp, offset = dp.align_radius(
            np.array(real_disp), dissim, gp_extrap=gp_extrap
        )

        offsets.append(offset)
        corrected_disps.append(corrected_disp)

        ax = the_figure.add_subplot(1, num_heights, subplot_num)
        # f.set_figheight(3)
        # f.set_figwidth(3)
        # plt.subplots(1,5,figsize=(15,15))
        # plt.subplots(figsize=(5, 5))

        # line_colour = (line_num  % num_angles) / num_angles

        plt.scatter(
            offset, 0, color=(line_colour, 0, 1 - line_colour), label=label, marker="+"
        )
        plt.axis([-11, 11, -3, 3])

        ax.annotate(
            int(angle),
            (offset, 1),
            fontsize=font_size - 3,
            ha="center",
            va="center",
            color="grey",
        )

        if line_num % num_angles == 1:

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

            # height = (subplot_num*0.5)-1.5
            # print(subplot_num)
            # print(1)
            # print(subplot_num-1)
            height = heights[subplot_num - 1]
            plt.title(
                f"Profiles at tap depth {height} mm from reference",
                fontsize=(font_size + 1),
            )

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

        if line_num <= num_angles:
            plt.legend(fontsize=font_size)

    if gp_extrap:
        full_path_png = os.path.join(
            data_home, current_experiment, "dissim_profiles_minimas_gp.png"
        )
        full_path_svg = os.path.join(
            data_home, current_experiment, "dissim_profiles_minimas_gp.svg"
        )
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)
        common.save_data(
            offsets, meta, name="post_processing/predicted_offsets_gp.json"
        )
        common.save_data(
            corrected_disps, meta, name="post_processing/corrected_disps_gp.json"
        )
    else:
        full_path_png = os.path.join(
            data_home, current_experiment, "dissim_profiles_minimas_basic.png"
        )
        full_path_svg = os.path.join(
            data_home, current_experiment, "dissim_profiles_minimas_basic.svg"
        )
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)
        common.save_data(
            offsets, meta, name="post_processing/predicted_offsets_basic.json"
        )
        common.save_data(
            corrected_disps, meta, name="post_processing/corrected_disps_basic.json"
        )
    if show_figs:
        plt.show()
    plt.clf()


def plot_height_flat(
    dissims, meta, data_home=None, current_experiment=None, show_fig=True
):
    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])
    num_disps = len(meta["line_range"])

    real_disp = meta["line_range"]
    real_heights = meta["height_range"]

    the_figure = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    print(f"dissims is shape {np.shape(dissims[1])}")

    reshaped_dissims = np.reshape(dissims, (num_angles * num_disps, num_heights))
    print(f"reshaped: {np.shape(reshaped_dissims)}")

    for i, profile in enumerate(reshaped_dissims):

        # for tap_num, _ in enumerate(dissims[1]):
        # if line_num < 19:
        #     angle = (line_num % num_angles) * 5 - 45  # todo, extract from meta
        #
        #     label = str(angle) + "°"
        # else:
        #     label = ""

        # if line_num % num_angles:
        #     line_colour = ((line_num) % num_angles) / num_angles
        # else:
        #     line_colour = 0

        # line_colour = ((tap_num) % num_angles) / num_angles

        # dissims = np.array(dissims)

        # for i in range(num_angles):
        #     print(i)
        #     line_colour = i / num_angles

        plt.plot(
            real_heights,
            # np.tile(real_heights, int(n_lines/(num_heights))),
            profile,
            # dissims[:,tap_num],
            # color=(line_colour, 0, 1 - line_colour),
            # label=label,
        )
    plt.legend()

    font_size = 10

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=font_size)

    # axis labels
    plt.xlabel("Height (mm)", fontsize=font_size, va="top")
    plt.ylabel("Dissimilarity", fontsize=font_size, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = parse_exp_name(exp_name[1])

    plt.gcf().text(
        0.01,
        1.01,
        f"At displacement={- np.array(meta['line_range'])}",
        transform=ax.transAxes,
        fontsize=font_size,
        alpha=0.2,
    )
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
        data_home, current_experiment, "dissim_height_profiles_keyed.png"
    )
    full_path_svg = os.path.join(
        data_home, current_experiment, "dissim_height_profiles_keyed.svg"
    )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_fig:
        plt.show()

    plt.clf()
    plt.close()


def plot_height_minimas(
    dissims, meta, data_home=None, current_experiment=None, gp_extrap=False
):
    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]
    num_disps = len(meta["line_range"])
    real_heights = np.array(meta["height_range"])

    the_figure = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    print(f"dissims is shape {np.shape(dissims[1])}")

    n_lines = len(dissims)
    # print(f"shape of dissims= {np.shape(dissims)} where dissim={dissims}")

    line_number = -1

    reshaped_dissims = dissims_height(dissims, meta)

    # for tap_num, _ in enumerate(dissims[1]):

    # if line_num < 19:
    #     angle = (line_num % num_angles) * 5 - 45  # todo, extract from meta
    #
    #     label = str(angle) + "°"
    # else:
    #     label = ""

    # if line_num % num_angles:
    #     line_colour = ((line_num) % num_angles) / num_angles
    # else:
    #     line_colour = 0

    # line_colour = ((tap_num) % num_angles) / num_angles

    # dissims = np.array(dissims)
    all_offsets = []

    for disp_index in range(num_disps):
        for angle_index in range(num_angles):
            line_number = (disp_index * num_angles) + angle_index
            print(f"line num: {line_number}")

            # print(i)
            line_colour = angle_index / num_angles
            # line_colour = disp_index / num_disps

            # height_dissim_profile = dissims[
            #     int(n_lines / (num_angles)) * angle : int(n_lines / (num_angles)) * (angle + 1),
            #     tap_num,
            # ]
            height_dissim_profile = reshaped_dissims[:, angle_index, disp_index]

            corrected_height, offset_height = dp.align_radius(
                real_heights,
                height_dissim_profile,
                gp_extrap=gp_extrap,
                start_hypers=[0.01, 100, 20],
                show_graph=False,
            )

            all_offsets.append(offset_height)

            plt.scatter(
                offset_height,
                line_number,
                marker="+",
                c=[line_colour, 0, 1 - line_colour],
            )

            # plt.plot(
            #     real_heights,
            #     # np.tile(real_heights, int(n_lines/(num_heights))),
            #     dissims[int(n_lines/(num_angles))*i:int(n_lines/(num_angles))*(i+1),tap_num],
            #     # dissims[:,tap_num],
            #     color=(line_colour, 0, 1 - line_colour),
            #     # label=label,
            # )
    plt.legend()

    plt.axis([-1.05, 1.05, 0, (num_disps * num_angles) + 1])

    # plot seperating section (by disp)
    for i in range(num_disps):
        height = i * num_angles
        plt.plot([-1, 1], [height - 0.5, height - 0.5], "k")
        plt.text(0.8, height + (num_angles / 2), f"disp = {real_disp[i]}mm")

    font_size = 10

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=font_size)

    # axis labels
    plt.xlabel("Height (mm)", fontsize=font_size, va="top")
    plt.ylabel("Tap number", fontsize=font_size, va="top")

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
    if gp_extrap:
        full_path_png = os.path.join(
            data_home, current_experiment, "dissim_height_minimas_gp.png"
        )
        full_path_svg = os.path.join(
            data_home, current_experiment, "dissim_height_minimas_gp.svg"
        )
    else:
        full_path_png = os.path.join(
            data_home, current_experiment, "dissim_height_minimas_simple.png"
        )
        full_path_svg = os.path.join(
            data_home, current_experiment, "dissim_height_minimas_simple.svg"
        )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    # plt.show()
    plt.clf()

    distributions = {
        "mean": np.mean(all_offsets),
        "abs_mean": np.mean(np.abs(all_offsets)),
        "perc_90": np.percentile(np.abs(all_offsets), 90),
        "perc_75": np.percentile(np.abs(all_offsets), 75),
        "perc_25": np.percentile(np.abs(all_offsets), 25),
    }

    if gp_extrap:
        common.save_data(
            distributions, meta, name="post_processing/distributions_gp.json"
        )
    else:
        common.save_data(
            distributions, meta, name="post_processing/distributions_simple.json"
        )
    print(all_offsets)
    print(f"mean = {np.mean(all_offsets)}")
    print(f"abs mean = {np.mean(np.abs(all_offsets))}")
    print(f"percentile 90  = {np.percentile(np.abs(all_offsets),90)}")
    print(f"percentile 75  = {np.percentile(np.abs(all_offsets),75)}")
    print(f"percentile 25  = {np.percentile(np.abs(all_offsets),25)}")


def get_height_minimas(dissims, meta, data_home=None, current_experiment=None):
    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])

    real_disp = meta["line_range"]
    num_disps = len(meta["line_range"])
    real_heights = np.array(meta["height_range"])

    the_figure = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    print(f"dissims is shape {np.shape(dissims[1])}")

    n_lines = len(dissims)
    # print(f"shape of dissims= {np.shape(dissims)} where dissim={dissims}")

    reshaped_dissims = np.reshape(dissims, (num_angles * num_disps, num_heights))
    print(f"reshaped: {np.shape(reshaped_dissims)}")

    a_profile = reshaped_dissims[1, :]
    print(a_profile)

    # for angle in range(num_angles):
    #     for disp in range(num_disps):
    #
    #         # print(height_profiles)
    #         # print(np.shape(reshaped_dissims[:,:,1]))
    #         # print(np.shape(reshaped_dissims[:,1,:]))
    #         print(np.shape(reshaped_dissims[:, angle, disp]))
    # # print(np.shape(reshaped_dissims[1,:,:]))

    for i, profile in enumerate(reshaped_dissims):

        corrected_height, offset_height = dp.align_radius(real_heights, profile)

        plt.scatter(offset_height, i, marker="+")

        # plt.plot(
        #     real_heights,
        #     # np.tile(real_heights, int(n_lines/(num_heights))),
        #     dissims[int(n_lines/(num_angles))*i:int(n_lines/(num_angles))*(i+1),tap_num],
        #     # dissims[:,tap_num],
        #     color=(line_colour, 0, 1 - line_colour),
        #     # label=label,
        # )
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
    plt.xlabel("Height (mm)", fontsize=font_size, va="top")
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
        data_home, current_experiment, "dissim_height_minimas.png"
    )
    full_path_svg = os.path.join(
        data_home, current_experiment, "dissim_height_minimas.svg"
    )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    plt.show()
    plt.clf()


# noinspection PyTypeChecker
def dissims_height(dissims, meta):
    # get dissim over height (as opposed to over displacement)

    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])
    num_disps = len(meta["line_range"])

    # reshaped_dissims = np.reshape(dissims.T, (num_angles * num_disps, num_heights))
    print(np.shape(dissims))
    reshaped_dissims = np.reshape(dissims, (num_heights, num_angles, num_disps))
    print(np.shape(reshaped_dissims))

    # noinspection PyTypeChecker
    # assert all(dissims[0] == reshaped_dissims[0,0])
    #
    # assert all(dissims[4] == reshaped_dissims[0,4])
    #
    # assert all(dissims[num_angles-1] == reshaped_dissims[0,num_angles-1])
    #
    # assert all(dissims[num_angles] == reshaped_dissims[1,0])
    #
    #
    # assert all(dissims[2*num_angles] == reshaped_dissims[2,0])
    #
    # assert all(dissims[(2*num_angles)+1] == reshaped_dissims[2,1])
    #
    # assert all(dissims[(num_heights*num_angles)-1] == reshaped_dissims[4,6])

    # flatten_again_dissim = np.reshape(reshaped_dissims, (num_heights, num_angles*num_disps)).T
    # # flatten_again_dissim = np.reshape(reshaped_dissims, (num_angles*num_disps,num_heights))
    #
    # print(np.shape(flatten_again_dissim))
    #
    # print()
    # assert all(reshaped_dissims[:,0,5] == flatten_again_dissim[5])
    # assert all(reshaped_dissims[:,1,0] == flatten_again_dissim[num_disps])

    return reshaped_dissims


def plot_seperate_angles(
    dissims, meta, data_home=None, current_experiment=None, show_fig=True
):
    num_heights = len(meta["height_range"])
    heights = meta["height_range"]
    num_angles = len(meta["angle_range"])
    angles = meta["angle_range"]
    num_disps = len(meta["line_range"])
    real_disp = meta["line_range"]
    dissims = np.array(dissims)

    # reshaped_dissims = np.reshape(dissims.T, (num_angles * num_disps, num_heights))
    reshaped_dissims = dissims_height(dissims, meta)

    the_figure = plt.figure(figsize=(20, 5))

    max_dissim = np.max(reshaped_dissims)
    min_dissim = np.min(reshaped_dissims)

    # for line_num, dissim in enumerate(reshaped_dissims):
    for disp_index in range(num_disps):
        for angle_index in range(num_angles):

            angle = angles[angle_index]
            # if -15 <= angle <= 15: # filter to show just this range
            print(angle)
            if disp_index == 0:  # line_num < num_angles:

                label = str(angle) + "°"

            else:
                label = ""

            # print(label)
            # if Y Z=X/Y else Z=0
            # z= ( x / y ) if y != 0 else 0
            # print(line_num)
            # if line_num % num_angles:
            line_colour = angle_index / num_angles
            # else:
            #     line_colour = 0

            # line_colour = ((line_num-1  % num_angles) / num_angles) if (line_num-1  % num_angles) != 0 else 0

            # print(line_colour)
            # if (line_num + 1) is 0:
            #     subplot_num = 1
            # else:
            #     subplot_num = int(np.ceil((line_num + 1) / num_angles))
            subplot_num = disp_index + 1
            ax = the_figure.add_subplot(3, 10, subplot_num)
            # f.set_figheight(3)
            # f.set_figwidth(3)
            # plt.subplots(1,5,figsize=(15,15))
            # plt.subplots(figsize=(5, 5))

            # line_colour = (line_num  % num_angles) / num_angles
            plt.plot(
                heights,
                reshaped_dissims[:, angle_index, disp_index],
                color=(line_colour, 0, 1 - line_colour),
                label=label,
            )
            plt.axis([-1.1, 1.1, min_dissim - 5, max_dissim + 5])

            font_size = 5
            if angle_index == 1:

                # Show the major grid lines with dark grey lines
                plt.grid(
                    b=True, which="major", color="#666666", linestyle="-", alpha=0.5
                )
                ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

                # Show the minor grid lines with very faint and almost transparent grey lines
                plt.minorticks_on()
                plt.grid(
                    b=True, which="minor", color="#999999", linestyle="-", alpha=0.2
                )
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

                # set axis font size
                plt.tick_params(labelsize=font_size)

                # axis labels
                plt.xlabel("Height (mm)", fontsize=font_size, va="top")
                plt.ylabel("Dissimilarity", fontsize=font_size, va="top")

                # height = (subplot_num*0.5)-1.5
                # print(subplot_num)
                # print(1)
                # print(subplot_num-1)
                disp = real_disp[subplot_num - 1]
                plt.title(
                    f"disp.={disp} mm from reference",
                    fontsize=(font_size + 1),
                )

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

            if disp_index == 0:
                plt.legend(fontsize=font_size)

    full_path_png = os.path.join(
        data_home, current_experiment, "dissim_profiles_angles.png"
    )
    full_path_svg = os.path.join(
        data_home, current_experiment, "dissim_profiles_angles.svg"
    )
    plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_fig:
        plt.show()
    plt.clf()


def main(
    ex, meta, data_home=None, current_experiment=None, show_figs=True, alt_ref=None
):

    neutral_tap = np.array(
        common.load_data(data_home + current_experiment + "neutral_tap.json")
    )
    if alt_ref is None:
        ref_tap = np.array(
            common.load_data(data_home + current_experiment + "ref_tap.json")
        )
    else:
        ref_tap = np.array(common.load_data(data_home + current_experiment + alt_ref))

    real_disp = meta["line_range"]  # nb, not copied so that reverse is persistent
    real_disp.reverse()  # to match previous works

    num_heights = len(meta["height_range"])
    num_angles = len(meta["angle_range"])
    num_disps = len(meta["line_range"])

    locations = []
    lines = []
    dissims = []
    num_lines = num_heights * num_angles

    # todo replace this with auto indexing files
    for line_num in range(1, num_lines + 1):

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
        dissims.append(dp.calc_dissims(np.array(best_frames), ref_tap, method="cosine"))

        common.save_data(locations, meta, name="post_processing/all_locations.json")
        common.save_data(lines, meta, name="post_processing/all_lines.json")
        if alt_ref is None:
            common.save_data(dissims, meta, name="post_processing/dissims.json")
        else:
            # replace alt_ref with dissims
            name_dissims = alt_ref.replace("ref_tap_", "dissims_")
            common.save_data(dissims, meta, name=name_dissims)

    # print(np.shape(dissims))
    # print(np.shape(dissims[0]))
    if np.shape(dissims) != (num_angles * num_heights, num_disps):
        print("Dataset incomplete, skipping processing")
    else:
        # plot_flat(dissims,meta)
        # plot_seperate_heights(
        #     dissims, meta, data_home=data_home, current_experiment=current_experiment
        # )
        plot_minimas(
            dissims,
            meta,
            gp_extrap=False,
            data_home=data_home,
            current_experiment=current_experiment,
            show_figs=False
        )
        # plot_height_flat(
        #     dissims,
        #     meta,
        #     data_home=data_home,
        #     current_experiment=current_experiment,
        #     show_fig=show_figs,
        # )
        # plot_seperate_angles(
        #     dissims,
        #     meta,
        #     data_home=data_home,
        #     current_experiment=current_experiment,
        #     show_fig=show_figs
        # )
        # plot_height_minimas(
        #     dissims, meta, data_home=data_home, current_experiment=current_experiment, gp_extrap=False
        # )
        # get_height_minimas(dissims, meta)

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
    # current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"
    # current_experiment = "collect_dataset_3d_21y-03m-26d_15h11m11s/"
    # current_experiment = "collect_dataset_3d_21y-03m-26d_15h17m30s/"
    # current_experiment = "collect_dataset_3d_21y-03m-26d_15h21m03s/"
    # current_experiment = "collect_dataset_3d_21y-03m-26d_15h25m15s/"
    # current_experiment = "collect_dataset_3d_21y-03m-26d_15h25m15s/"
    # current_experiment = "collect_dataset_3d_21y-03m-26d_15h33m28s/"

    # Profile at -5 disp (ie, off edge)
    # current_experiment = "collect_dataset_3d_21y-03m-29d_11h24m47s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_11h26m58s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_11h28m13s/"

    # Profile at 0
    # current_experiment = "collect_dataset_3d_21y-03m-29d_11h30m44s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_11h37m45s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h09m01s/"

    # Profile at 5 disp (ie on object)
    # current_experiment = "collect_dataset_3d_21y-03m-29d_11h39m32s/"

    # Profile at -1
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h02m02s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h10m26s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h13m30s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h14m25s/"

    # Profile at -2
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h03m28s/"
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h04m40s/"

    # Profile at -3
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h05m59s/"

    # Profile at -4
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h07m32s/"

    # Profile at 1
    # current_experiment = "collect_dataset_3d_21y-03m-29d_12h12m18s/"

    # complete set
    # current_experiment = "collect_dataset_3d_21y-03m-30d_12h06m43s/"
    # current_experiment = "collect_dataset_3d_21y-04m-13d_14h57m22s/"
    # current_experiment = "collect_dataset_3d_21y-04m-19d_11h18m10s/"
    # current_experiment = "collect_dataset_3d_21y-04m-20d_13h47m03s/"
    # current_experiment = "collect_dataset_3d_21y-04m-20d_14h43m12s/"
    # current_experiment = "collect_dataset_3d_21y-11m-19d_12h24m42s/"
    # current_experiment = "collect_dataset_3d_21y-11m-22d_16h10m54s/"

    # current_experiment = "collect_dataset_3d_21y-12m-07d_16h00m01s/"
    # current_experiment = "collect_dataset_3d_21y-12m-07d_15h24m32s/"
    current_experiment = "collect_dataset_3d_21y-12m-07d_12h33m47s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(
        state.ex, state.meta, data_home=data_home, current_experiment=current_experiment
    )
