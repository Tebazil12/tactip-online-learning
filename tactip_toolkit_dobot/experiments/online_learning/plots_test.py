import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Wedge, Rectangle
from mpl_toolkits.mplot3d import Axes3D

import tactip_toolkit_dobot.experiments.online_learning.contour_following_3d as online


def plot_all_movements_basic_test(ex, meta, show_figs=True, save_figs=True):
    print("hello")
    all_edge_np = np.array(ex.edge_locations)

    pos_ys_e = all_edge_np[:, 0]
    pos_xs_e = all_edge_np[:, 1]

    pos_xs2 = all_edge_np[:, 0]
    pos_ys2 = all_edge_np[:, 1]
    heights2 = ex.edge_height

    # plot_0_size = (max(pos_ys_e)+10) - (min(pos_ys_e)-10)
    # plot_1_size = (max(heights2)+10) - (min(heights2)-10)
    plot_0_size = (max(pos_ys_e)) - (min(pos_ys_e))
    plot_1_size = (max(heights2)) - (min(heights2))

    if plot_0_size < 5:
        plot_0_size = 5
    if plot_1_size < 5:
        plot_1_size = 5

    plot_width = (max(pos_xs_e)) - (min(pos_xs_e))

    print(pos_ys_e)
    print(heights2)

    print(f" size 0 {plot_0_size} soze 1 {plot_1_size}")

    print(f"here {plt.rcParamsDefault['figure.figsize']}")
    plt.rcParams["figure.figsize"] = (
        (plot_width + 20) / 10,
        ((plot_0_size + plot_1_size) + 20) / 10,
    )
    print(f"here2 {plt.rcParams['figure.figsize']}")

    fig, ax = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [plot_0_size, plot_1_size]}
    )
    fig.subplots_adjust(hspace=0.01)

    ax[0].plot(
        pos_xs_e,
        pos_ys_e,
        color="#FFAA00",  # "#711CFC",
        marker="",
        markersize=6,
        linewidth=3,
        linestyle="-",
    )

    ax[1].plot(
        pos_ys2,
        heights2,
        color="#FFAA00",  # "#15b01a",
        marker="",
        markersize=6,
        linewidth=3,
        linestyle="-",
    )

    # Show the major grid lines with dark grey lines
    ax[1].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    ax[1].minorticks_on()
    ax[1].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # Show the major grid lines with dark grey lines
    ax[0].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    ax[0].minorticks_on()
    ax[0].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].set_aspect("equal", adjustable="datalim")
    # ax[1].autoscale()

    # ax[0].axis([-10, 100,  max(pos_ys_e)+10, min(pos_ys_e)-10])

    # ax[0].set_ylim([-10,50])

    ax[1].set_xlim([-10, 100])
    ax[0].set_xlim([-10, 100])
    #
    # ax[1].set_ylim([-5,5])
    #
    plt.show()


def plot_all_movements_both(ex, meta, show_figs=True, save_figs=True):

    # load data for edge locations
    all_edge_np = np.array(ex.edge_locations)

    robots_xs = all_edge_np[:, 0]
    robots_ys = all_edge_np[:, 1]
    robots_heights = ex.edge_height

    # plot_0_size = (max(pos_ys_e)+10) - (min(pos_ys_e)-10)
    # plot_1_size = (max(heights2)+10) - (min(heights2)-10)
    plot_0_size = (max(robots_xs)) - (min(robots_xs))
    plot_1_size = (max(robots_heights)) - (min(robots_heights))

    plot_width = (max(robots_ys)) - (min(robots_ys))

    if plot_0_size < 5:
        plot_0_size = 5
    if plot_1_size < 5:
        plot_1_size = 5

    print(f"here {plt.rcParamsDefault['figure.figsize']}")
    plt.rcParams["figure.figsize"] = (
        (plot_width + 20) / 15,
        ((plot_0_size + plot_1_size) + 20) / 15,
    )
    print(f"here2 {plt.rcParams['figure.figsize']}")

    # fig = plt.figure()
    # fig, ax = plt.subplots(2,sharex=True)
    # fig.subplots_adjust(hspace=0)
    fig, ax = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [plot_0_size, plot_1_size]}
    )
    fig.subplots_adjust(hspace=0.01)

    ax[1].set_xlim([-10, 100])
    ax[0].set_xlim([-10, 100])

    line_width = 1.5
    marker_size = 1
    # ax = plt.gca()
    # ax = fig.add_axes([0,0,1,1])

    if meta["stimuli_name"] == "70mm-circle":
        # print small circle location
        radius = 35
        x_offset = 35 - 35
        y_offset = 0 + 35
        # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
        ang = np.linspace(np.pi / 2, -np.pi / 2, 100)
        x = x_offset + radius * -np.cos(ang)
        y = y_offset + radius * np.sin(ang)
        ax[0].plot(x, y, "tab:brown", linewidth=line_width)
        # y=y*.8
        # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')

        # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
        w2 = Wedge((x_offset, y_offset), radius, 90, -90, fc="tab:brown", alpha=0.5)
        ax[0].add_artist(w2)
    elif meta["stimuli_name"] == "105mm-circle":
        # print large circle location
        radius = 107.5 / 2
        x_offset = radius
        y_offset = 0
        # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
        ang = np.linspace(np.pi / 2, -np.pi / 2, 100)
        x = x_offset + radius * -np.cos(ang)
        y = y_offset + radius * np.sin(ang)
        ax[0].plot(x, y, "tab:brown", linewidth=line_width)
        # y=y*.8
        # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')

        # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
        w2 = Wedge((x_offset, y_offset), radius, 90, -90, fc="tab:brown", alpha=0.5)
        ax[0].add_artist(w2)

    # elif meta["stimuli_name"] == "flower":
    #     img = plt.imread("/home/lizzie/Pictures/stimulus-flower.png")
    #     img_cropped = img[:, 0 : int(img.shape[0] / 2), :]
    #     f_size = 126
    #     f_y_offset = -5.2
    #     ax[0].imshow(
    #         img_cropped,
    #         extent=[-f_size / 2, 0, 0 + f_y_offset, f_size + f_y_offset],
    #         alpha=0.5,
    #     )
    elif meta["stimuli_name"] == "flower":
        img = plt.imread("/home/lizzie/Pictures/stimulus-flower2.png")
        img_cropped = img[:, :, 0 : int(img.shape[0] / 2)]
        f_size = 126
        f_y_offset = 0  # -5.2
        ax[0].imshow(
            img_cropped,
            extent=[
                0,
                f_size,
                -f_size / 2 + f_y_offset,
                f_size / 2 + f_y_offset,
            ],  # (left, right, bottom, top)
            alpha=0.5,
        )

    elif meta["stimuli_name"] == "banana-screwed":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/banana-top-2.jpg"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 150
        desired_width = int(img_width / (img_height / desired_height))

        desired_y_offset = -78
        desired_x_offset = -78
        ax[0].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                desired_height + desired_y_offset,
                0 + desired_y_offset,
            ],
            alpha=0.6,
        )

    elif meta["stimuli_name"] == "cap-mid":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/cap-above.jpg"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 150 * (5 / 15.7) * (125 / 35) * (125 / 127)
        desired_width = int(img_width / (img_height / desired_height))

        desired_y_offset = -23.5 - 55 + 10 - 1
        desired_x_offset = -50 - 7 + 1 - 15 + 1
        ax[0].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                desired_height + desired_y_offset,
                0 + desired_y_offset,
            ],
            alpha=0.8,
        )

    elif meta["stimuli_name"] == "balance-lid" or meta["stimuli_name"] == "lid-screwed":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/lid-above.jpg"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 150
        desired_width = int(img_width / (img_height / desired_height))

        desired_y_offset = -23.5 - 55 + 10 - 1 + 30 + 7 - 2 - 1
        desired_x_offset = -50 - 7 + 1 - 15 + 1 - 4.5 + 5
        ax[0].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                desired_height + desired_y_offset,
                0 + desired_y_offset,
            ],
            alpha=1,
        )

    elif meta["stimuli_name"] == "wavy-line-thin":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/wave-2d-2.png"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 148.2
        desired_width = img_width / (img_height / desired_height)

        desired_y_offset = -23.5 - 67.3
        desired_x_offset = -50 - 7 + 1 - 30 + 4.5 - 0.1 + 2 - 5
        ax[0].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                desired_height + desired_y_offset,
                0 + desired_y_offset,
            ],
            alpha=0.3,
        )

    elif meta["stimuli_name"] == "wavy-line-thick":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/wave-2d-2.png"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 148.2
        desired_width = img_width / (img_height / desired_height)

        desired_y_offset = -23.5 - 67.3 + 47.5
        desired_x_offset = -50 - 7 + 1 - 30 + 4.5 - 0.1 + 2 - 5
        ax[0].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                desired_height + desired_y_offset,
                0 + desired_y_offset,
            ],
            alpha=0.3,
        )

    elif meta["stimuli_name"] == "wavy-edge-3d":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/wave-3d-top.png"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = (
            150 * 1.7 * (60 / 64.7) * (100 / 202) * (100 / (34.5 + 65)) * 1.2
        )
        desired_width = int((img_width / (img_height / desired_height)))

        desired_y_offset = -83 - 0.5 + 6 + 44 - 1.5
        desired_x_offset = -60 + 3 + 7 + 11 + 5
        ax[0].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                desired_height + desired_y_offset,
                0 + desired_y_offset,
            ],
            alpha=0.6,
        )
    elif meta["stimuli_name"] == "saddle-high":
        # print large circle location
        radius = 107.5 / 2
        x_offset = 0
        y_offset = radius + 2
        # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
        ang = np.linspace(np.pi, -np.pi, 100)
        x = x_offset + radius * -np.cos(ang)
        y = y_offset + radius * np.sin(ang)
        ax[0].plot(x, y, "tab:brown", linewidth=line_width)
        # y=y*.8
        # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')

        # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
        w2 = Wedge((x_offset, y_offset), radius, 180, -180, fc="tab:brown", alpha=0.5)
        ax[0].add_artist(w2)

    elif meta["stimuli_name"].split("-")[0] == "tilt":
        # plt.plot([0, 80, 80], [0, 0, 100])
        ax[0].fill([-10, 100, 100, -10], [0, 0, 100, 100], "grey", alpha=0.6)

    if False:
        # print all tap locations
        all_tap_positions_np = np.array(ex.all_tap_positions)
        pos_xs = all_tap_positions_np[2:, 0]
        pos_ys = all_tap_positions_np[2:, 1]
        # pos_ys = pos_ys/0.8
        n = range(len(pos_xs))
        ax[0].plot(
            pos_ys,
            pos_xs,
            "k",
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
        )
        # plt.scatter(pos_xs, pos_ys, color="k", s=marker_size)

        [
            ax[0].annotate(
                int(x[0]),
                (x[1], x[2]),
                fontsize=1,
                ha="center",
                va="center",
                color="grey",
            )
            for x in np.array([n, pos_ys, pos_xs]).T
        ]

        # print data collection lines
        for line in ex.line_locations:
            line_locations_np = np.array(line)
            ax[0].plot(
                line_locations_np[:, 1],
                line_locations_np[:, 0],
                "r",
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
            )
            # plt.scatter(line_locations_np[:, 0], line_locations_np[:, 1], color="g",s=marker_size)

    if ex.edge_locations is not None:
        # print predicted edge locations

        # if meta["stimuli_name"] == "balance-lid":
        #     pos_xs_e = pos_xs_e * 0.84

        # pos_ys = pos_ys/0.8
        n = range(len(robots_ys))
        if meta["stimuli_name"] == "wavy-edge-3d":
            line_style = "solid"
        elif meta["stimuli_name"] == "wavy-raised-3d":
            line_style = (0, (5, 1))
        elif meta["stimuli_name"] == "wavy-line-thin-3d":
            line_style = (0, (1, 1))
        else:
            line_style = "solid"
        if meta["plane_method"] == "cross":
            # pass
            ax[0].plot(
                robots_ys,
                robots_xs,
                color="#FFAA00",  # "#711CFC",
                marker="",
                markersize=marker_size + 1,
                linewidth=line_width,
                linestyle=line_style,
            )
        else:
            ax[0].plot(
                robots_ys,
                robots_xs,
                color="#30E641",  # "#15b01a",
                marker="",
                markersize=marker_size + 1,
                linewidth=line_width,
                linestyle=line_style,
            )
    # plt.scatter(pos_xs, pos_ys, color="r",marker='+',s=marker_size)
    ax[0].set_aspect("equal", adjustable="datalim")

    # Show the major grid lines with dark grey lines
    ax[0].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    ax[0].minorticks_on()
    ax[0].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    font_size = 8
    ax[0].tick_params(labelsize=font_size)

    # axis labels
    plt.xlabel("y displacement (mm)", fontsize=font_size, va="top")
    plt.ylabel("x displacement (mm)", fontsize=font_size, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = online.parse_exp_name(exp_name[1])

    plt.gcf().text(
        0.01,
        1.01,
        meta["stimuli_name"],
        transform=ax[0].transAxes,
        fontsize=4,
        alpha=0.2,
    )
    plt.gcf().text(
        1,
        1.01,
        readable_name,
        transform=ax[0].transAxes,
        fontsize=4,
        alpha=0.2,
        ha="right",
    )
    #     # Don't allow the axis to be on top of your data
    ax[0].set_axisbelow(True)

    # ax[0].set(auto=True)
    # xmin, xmax, ymin, ymax = plt.axis()
    # print(xmax)
    # plt.axis([xmin, xmax + 2, ymin, ymax])

    if meta["stimuli_name"] == "banana-screwed":
        plt.axis(
            [
                min(robots_ys) - 10,
                max(robots_ys) + 10,
                max(robots_xs) + 20,
                min(robots_xs) - 20,
            ]
        )
    elif (
        meta["stimuli_name"] == "wavy-line-thin"
        or meta["stimuli_name"] == "wavy-line-thick"
    ):

        # ax[0].axis([-10, 100,  max(pos_ys_e)+10, min(pos_ys_e)-10])
        # ax[0].set_ylim([max(pos_ys_e)+10, min(pos_ys_e)-10])
        pass

    elif (
        meta["stimuli_name"].split("-")[0] == "wavy"
        and meta["stimuli_name"].split("-")[-1] == "3d"
    ):
        plt.axis([-10, 100, 80 - 5 - 2, -10 + 5 - 2])
    elif meta["stimuli_name"].split("-")[0] == "tilt":
        plt.axis([-2, 40, 5, -5])
    elif meta["stimuli_name"] == "cap-mid":
        plt.axis([-5, 65, 10, -15])
    elif meta["stimuli_name"] == "lid-screwed":
        plt.axis([-5, 45, 45 + 2, -5 + 2])
    else:
        plt.axis(
            [
                min(robots_ys) - 1,
                max(robots_ys) + 1,
                max(robots_xs) + 1,
                min(robots_xs) - 1,
            ]
        )

    ##### 3d stuff #####
    #
    #
    # fig = plt.figure()
    #
    # line_width = 1.5
    # marker_size = 1
    # ax = plt.gca()
    # ax = fig.add_axes([0,0,1,1])
    if meta["stimuli_name"] == "70mm-circle":
        # print small circle location
        radius = 35
        x_offset = 35 - 35
        y_offset = 0 + 35
        # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
        ang = np.linspace(np.pi / 2, -np.pi / 2, 100)
        x = x_offset + radius * -np.cos(ang)
        y = y_offset + radius * np.sin(ang)
        ax[1].plot(x, y, "tab:brown", linewidth=line_width)
        # y=y*.8
        # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')

        # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
        w2 = Wedge((x_offset, y_offset), radius, 90, -90, fc="tab:brown", alpha=0.5)
        ax[1].add_artist(w2)
    elif meta["stimuli_name"] == "105mm-circle":
        # print large circle location
        radius = 50
        x_offset = -radius
        y_offset = 0
        # --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles
        ang = np.linspace(np.pi / 2, -np.pi / 2, 100)
        x = x_offset + radius * -np.cos(ang)
        y = y_offset + radius * np.sin(ang)
        # plt.plot(x, y, "tab:brown", linewidth=line_width)
        # y=y*.8
        # plt.plot(x, y,'tab:brown',linewidth=line_width, linestyle='dashed')

        # Arc(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs
        # w2 = Wedge((x_offset, y_offset), radius, 90, -90, fc="tab:brown", alpha=0.5)
        w2 = Rectangle((x_offset, y_offset), radius * 2, -10, fc="tab:brown", alpha=0.5)
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

    elif meta["stimuli_name"] == "banana-screwed":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/banana-side-2.jpg"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        desired_width = 300
        desired_height = int(img_height / (img_width / desired_width))
        desired_y_offset = -52 - 10 - 30 - 1
        desired_x_offset = -10 - 70 - 40 + 5 + 5 - 1.5
        ax.imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                0 + desired_y_offset,
                desired_height + desired_y_offset,
            ],
            alpha=0.6,
        )

    elif meta["stimuli_name"] == "wavy-edge-3d":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/wave-side-reversed.png"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = (
            150 * 1.7 * (5 / 14) * (120 / (103 + 18.25)) * (120 / (162 + 24.6))
        )
        desired_width = int((img_width / (img_height / desired_height)))

        desired_y_offset = -83 - 0.5 + 46.5 - 5 + 21 - 5
        desired_x_offset = -60 + 6 + 10 - 2 + 3 - 2.5 + 1 + 3.4 - 3 + 1.6 + 15
        ax.imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                0 + desired_y_offset,
                desired_height + desired_y_offset,
            ],
            alpha=0.6,
        )

    elif meta["stimuli_name"] == "cap-mid":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/cap-side.jpg"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = (
            150
            * 1.7
            * (5 / 14)
            * (120 / (103 + 18.25))
            * (120 / (162 + 24.6))
            * (125 / 94.7)
        )
        desired_width = int((img_width / (img_height / desired_height)))

        desired_y_offset = -83 - 0.5 + 46.5 - 5 + 21 - 5 - 15 - 3
        desired_x_offset = (
            -60 + 6 + 10 - 2 + 3 - 2.5 + 1 + 3.4 - 3 + 1.6 + 15 - 40 - 12 + 1.5
        )
        ax.imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                0 + desired_y_offset,
                desired_height + desired_y_offset,
            ],
            alpha=0.8,
        )

    elif meta["stimuli_name"] == "lid-screwed":
        img = plt.imread(
            "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/icra2023/lid-side.jpg"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 150 * (76 / 179)
        desired_width = int((img_width / (img_height / desired_height)))

        desired_y_offset = -44.5
        desired_x_offset = -52 + 3
        ax.imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                0 + desired_y_offset,
                desired_height + desired_y_offset,
            ],
            alpha=1,
        )

    elif meta["stimuli_name"] == "saddle-high":

        img = plt.imread(
            "/home/lizzie/Pictures/Screenshot from 2022-12-06 15-24-58.png"
        )
        img_cropped = img  # [:, 0 : int(img.shape[0] / 2), :]

        print(f"image is size {img.shape}")

        img_width = img.shape[1]
        img_height = img.shape[0]

        # desired_width = 150 *0.02639
        # desired_height = int(img_height / (img_width/desired_width))
        desired_height = 80
        desired_width = int((img_width / (img_height / desired_height)))

        desired_y_offset = -83 - 0.5 + 46.5 - 5 + 21 - 5 - 32.3
        desired_x_offset = -60 + 6 + 10 - 2 + 3 - 2.5 + 1 + 3.4 - 3 + 1.6 + 15 - 45
        ax[1].imshow(
            img_cropped,
            extent=[
                desired_x_offset + 0,
                desired_x_offset + desired_width,
                0 + desired_y_offset,
                desired_height + desired_y_offset,
            ],
            alpha=0.6,
        )

    elif meta["stimuli_name"].split("-")[0] == "tilt":
        pass

    else:
        ax[1].fill([-10, 100, 100, -10], [0, 0, -100, -100], "grey", alpha=0.6)

    if meta["stimuli_name"] == "tilt-05deg-down":
        ax[1].plot([0, 100], [0, -8.7], ":k")
    elif meta["stimuli_name"] == "tilt-10deg-down":
        ax[1].plot([0, 100], [0, -17.6], ":k")
    elif meta["stimuli_name"] == "tilt-20deg-down":
        ax[1].plot([0, 100], [0, -36.4], ":k")
    elif meta["stimuli_name"] == "tilt-05deg-up":
        ax[1].plot([0, 100], [0, 8.7], ":k")
    elif meta["stimuli_name"] == "tilt-10deg-up":
        ax[1].plot([0, 100], [0, 17.6], ":k")
    elif meta["stimuli_name"] == "tilt-20deg-up":
        ax[1].plot([0, 100], [0, 36.4], ":k")
    elif meta["stimuli_name"] == "tilt-0deg":
        ax[1].plot([0, 100], [0, 0], ":k")

    if False:
        # print all tap locations
        all_tap_positions_np = np.array(ex.all_tap_positions)
        pos_xs = all_tap_positions_np[2:, 0]  # remove ref and neutral taps
        pos_ys = all_tap_positions_np[2:, 1]
        heights = all_tap_positions_np[2:, 3]
        # pos_ys = pos_ys/0.8
        n = range(len(pos_xs))
        ax[1].plot(
            pos_ys,
            heights,
            "k",
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
        )
        # plt.scatter(pos_xs, pos_ys, color="k", s=marker_size)

        [
            ax.annotate(
                int(x[0]),
                (x[1], x[2]),
                fontsize=1,
                ha="center",
                va="center",
                color="grey",
            )
            for x in np.array([n, pos_ys, heights]).T
        ]

    if ex.edge_locations is not None:
        # print predicted edge locations

        # pos_ys = pos_ys/0.8
        n = range(len(robots_xs))

        if meta["stimuli_name"] == "wavy-edge-3d":
            line_style = "solid"
        elif meta["stimuli_name"] == "wavy-raised-3d":
            line_style = (0, (5, 1))
            robots_heights = robots_heights + 2
        elif meta["stimuli_name"] == "wavy-line-thin-3d":
            line_style = (0, (1, 1))
        else:
            line_style = "solid"

        if meta["plane_method"] == "cross":
            # pass
            ax[1].plot(
                robots_ys,
                robots_heights,
                color="#FFAA00",  # "#15b01a",
                marker="",
                markersize=marker_size + 1,
                linewidth=line_width,
                linestyle=line_style,
            )
        else:
            ax[1].plot(
                robots_ys,
                robots_heights,
                color="#30E641",  # "#15b01a",
                marker="",
                markersize=marker_size + 1,
                linewidth=line_width,
                linestyle=line_style,
            )

    # plt.scatter(pos_xs, pos_ys, color="r",marker='+',s=marker_size)
    ax[1].set_aspect("equal", adjustable="datalim")

    # Show the major grid lines with dark grey lines
    ax[1].grid(b=True, which="major", color="#666666", linestyle="-", alpha=0.5)
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Show the minor grid lines with very faint and almost transparent grey lines
    ax[1].minorticks_on()
    ax[1].grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # set axis font size
    plt.tick_params(labelsize=8)

    # axis labels
    plt.xlabel("y displacement (mm)", fontsize=8, va="top")
    plt.ylabel("height (mm)", fontsize=8, va="top")

    # add identifier labels
    part_path, _ = os.path.split(meta["meta_file"])

    exp_name = part_path.split("/")
    readable_name = online.parse_exp_name(exp_name[1])

    plt.gcf().text(
        0.01,
        1.01,
        meta["stimuli_name"],
        transform=ax[1].transAxes,
        fontsize=4,
        alpha=0.2,
    )
    plt.gcf().text(
        1,
        1.01,
        readable_name,
        transform=ax[1].transAxes,
        fontsize=4,
        alpha=0.2,
        ha="right",
    )
    #     # Don't allow the axis to be on top of your data
    ax[1].set_axisbelow(True)

    # ax.set(auto=True)
    # xmin, xmax, ymin, ymax = plt.axis()
    # print(xmax)
    # plt.axis([xmin, xmax + 2, ymin, ymax])

    if meta["stimuli_name"] == "banana-screwed":
        ax[1].axis(
            [
                min(robots_ys) - 10,
                max(robots_ys) + 10,
                min(robots_heights) - 10,
                max(robots_heights) + 10,
            ]
        )
    elif (
        meta["stimuli_name"].split("-")[0] == "wavy"
        and meta["stimuli_name"].split("-")[-1] == "3d"
    ):
        ax[1].axis([-10, 100, -5 + 3, 15 + 3])
    elif (
        meta["stimuli_name"] == "wavy-line-thin"
        or meta["stimuli_name"] == "wavy-line-thick"
    ):

        # ax[1].axis([ -10, 100, -5, 5])
        # ax[1].set_ylim([ -5, 5])
        pass

    elif meta["stimuli_name"].split("-")[0] == "tilt":
        ax[1].axis([-2, 40, -10, 20])
    elif meta["stimuli_name"] == "cap-mid":
        ax[1].axis([-5, 65, -15, 5])
    elif meta["stimuli_name"] == "lid-screwed":
        ax[1].axis([-5, 45, -12, 2])
    else:
        ax[1].axis(
            [
                min(robots_ys) - 1,
                max(robots_ys) + 1,
                min(robots_heights) - 1,
                max(robots_heights) + 1,
            ]
        )
        # plt.axis([min(pos_ys) -1, max(pos_ys) +1, min(heights) -1, max(heights)+1])

    #### end 3d stuff ####

    if save_figs:
        # save graphs automatically
        part_path, _ = os.path.split(meta["meta_file"])
        full_path_png = os.path.join(
            meta["home_dir"],
            part_path,
            "all_movements_final_both-"
            + meta["stimuli_name"]
            + "-"
            + meta["plane_method"]
            + ".png",
        )
        full_path_svg = os.path.join(
            meta["home_dir"],
            part_path,
            "all_movements_final_both-"
            + meta["stimuli_name"]
            + "-"
            + meta["plane_method"]
            + ".svg",
        )
        plt.savefig(full_path_png, bbox_inches="tight", pad_inches=0, dpi=1000)
        plt.savefig(full_path_svg, bbox_inches="tight", pad_inches=0)

    if show_figs:
        plt.show()

    if show_figs or save_figs:
        plt.clf()
        plt.close()
