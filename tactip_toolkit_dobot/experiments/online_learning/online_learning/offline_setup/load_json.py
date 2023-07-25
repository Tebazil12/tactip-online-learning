import json
import numpy as np


def load_data():
    # with open('/home/lizzie/OneDrive/data/collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901/c45_01_20.json') as f:
    with open(
        "C:\\Users\\ea-stone\\Documents\\data\\collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901\\all_data.json"
    ) as f:
        data = json.load(f)
    # print(data[1][1][1])
    # print(type(data))
    # print(len(data))
    ##print(data[1])
    # print(type(data[1]))
    # print(len(data[1]))
    # print(len(data[1][1]))
    # print(len(data[1][1][1]))

    # all_data = np.array(data)

    # data[tap_number][frame][pin][xorydisp]
    n_disps = 21
    n_angles = 19
    n_depths = 9
    n_radii = n_angles * n_depths

    all_data = [[[None] * n_disps] * n_angles] * n_depths

    current_number = 0
    for depth in range(0, n_depths):
        for angle in range(0, n_angles):
            for disp in range(0, n_disps):
                # make list of arrays
                #            print(current_number)
                all_data[depth][angle][disp] = np.array(data[current_number])
                current_number = current_number + 1

    # n_taps = 129
    # n_frames = 16 #ish
    # n_pins = 125#ish
    #
    # current_number = 0
    # for tap_num in range(0,n_taps):
    #    for angle in range(0,n_angles):
    #        for disp in range(0,n_disps):
    #            # make list of arrays
    #            all_data[depth][angle][disp]= np.array(data[current_number])
    #            current_number = current_number + 1

    return all_data


def whatever(all_data):
    n_disps = 21
    n_angles = 19
    n_depths = 9
    n_radii = n_angles * n_depths
    x_real = np.empty([n_radii], dtype=object)
    x_real[0] = np.arange(-10, 11)  # actually -10 to 10 but +1 cuz python
    # print(x_real)

    x_real_test = np.arange(-10, 11)  # actually -10 to 10 but +1 cuz python
    X_SHIFT_ON = False

    ## Define reference tap & stuff
    print("~~~~~~~~Define reference tap & stuff")

    print(type(all_data[5]))
    ref_tap = all_data[5][10][11]
    # print(ref_tap)
    print(type(ref_tap))
    print("ref_tap[1]")
    print(type(ref_tap[1]))

    # Normalize data, so get distance moved not just relative position
    ref_diffs_norm = (
        ref_tap - ref_tap[1]
    )  # normalized, assumes starts on no contact/all start in same position

    # find the frame in ref_diffs_norm with greatest diffs
    max_indexes = np.argmax(abs(ref_diffs_norm), 0)

    mean_max_index = round(np.mean(max_indexes))

    dissims = []
    return ref_diffs_norm


if __name__ == "__main__":
    pass
