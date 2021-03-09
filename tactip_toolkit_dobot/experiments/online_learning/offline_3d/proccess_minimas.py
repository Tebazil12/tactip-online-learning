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

def main(ex, meta):
    data = common.load_data(data_home + current_experiment+ "predicted_offsets.json")

    data_np = np.array(data)

    data_cleaned = data_np[ (data_np <= 9) & (data_np >= -9)]


    data_mean = np.mean(data_cleaned)
    data_std = np.std(data_cleaned)
    q1 = np.percentile(data_cleaned, 25, interpolation = 'midpoint')
    # q1 = np.mean(data_cleaned[:halfway])
    # q3 = np.mean(data_cleaned[halfway:])
    q3 = np.percentile(data_cleaned, 75, interpolation = 'midpoint')
    p5 = np.percentile(data_cleaned, 5, interpolation = 'midpoint')
    p95 = np.percentile(data_cleaned, 95, interpolation = 'midpoint')

    print(f"stats --- mean: {np.round(data_mean,3)} st.dev.: {np.round(data_std,3)} Q1: {np.round(q1,3)} Q3: {np.round(q3,3)} P5: {np.round(p5,3)} P95: {np.round(p95,3)}")

if __name__ == "__main__":

    data_home = (
        "/home/lizzie/git/tactip_toolkit_dobot/data/TacTip_dobot/online_learning/"
    )
    current_experiment = "collect_dataset_3d_21y-03m-03d_15h18m06s/"

    state = State(meta=common.load_data(data_home + current_experiment + "meta.json"))

    print(state.meta["stimuli_name"])

    state.ex = Experiment()
    main(state.ex, state.meta)
