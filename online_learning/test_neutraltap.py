import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt

import tactip_toolkit_dobot.experiments.min_example.common as common
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
)


np.set_printoptions(precision=2, suppress=True)


def main():
    meta = make_meta()
    ex = Experiment()

    with common.make_robot() as ex.robot, common.make_sensor(meta) as ex.sensor:
        # print("now clearing")
        # results = ex.robot.sync_robot.clearAlarms()

        print("initing robot")
        common.init_robot(ex.robot, meta, do_homing=False)

        print("Main code...")

        # collect neutral, non-contact position (as reference for other taps)
        neutral_tap, keypoints = ex.processed_tap_at(
            [-20, -80], 0, meta, selection_criteria="Mean"
        )  # tODO, rework taps so can set z # TODO tap motion is not needed

        print("collected:")
        print(neutral_tap)

        common.go_home(ex.robot, meta)

        # # save data
        # common.save_data(ref_tap, meta)

        # show plot of tap taken (note, this pauses program until graph is closed)
        for frame in keypoints:
            plt.scatter(frame[:, 0], frame[:, 1])
        plt.axis("equal")
        plt.show()

        # show plot of tap taken (note, this pauses program until graph is closed)
        ref_tap = np.array(neutral_tap)
        ref_tap = ref_tap.reshape(int(ref_tap.shape[0] / 2), 2)
        print(ref_tap)
        plt.scatter(ref_tap[:, 0], ref_tap[:, 1])
        plt.axis("equal")
        plt.show()

    print("Done, exiting")


if __name__ == "__main__":
    main()
