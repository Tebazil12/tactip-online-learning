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

        ex.collect_neutral_tap(meta)
        all_results = None
        num_reps = 10
        results = [None] * num_reps

        for i in range(0,num_reps):
            results[i] = ex.collect_line([0, 0], np.deg2rad(90), meta)
            if all_results is None:
                all_results = results[i]
            else:
                all_results = np.vstack((all_results,results[i]))

        print("collected:")
        print(all_results)

        # print("raw data")
        # print(ex.all_raw_data)

        common.go_home(ex.robot, meta)

        # save data
        results_np = np.array(all_results)  # needed so can json.dump properly
        common.save_data(results_np.tolist(), meta)

        print(f"the mean std of collected data = {np.mean(np.std(results,axis=0))}")

    print("Done, exiting")
 # np.shape(np.mean(np.max(abs(data),axis=1),axis=0))

if __name__ == "__main__":
    main()
