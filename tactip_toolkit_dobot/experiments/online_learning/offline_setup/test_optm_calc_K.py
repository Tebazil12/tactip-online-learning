import time

import numpy as np
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.gp as gp
import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

def calc_K_faster(x, sigma_f, L, sigma_n):
    print("in calc_K")
    print(f"\n a {time.perf_counter()}")

    n_xs = len(x)
    print(f"\n b {time.perf_counter()}")
    k_cap = np.zeros((n_xs, n_xs))
    # print(f"k_cap here = {k_cap}")
    print(f"\n c {time.perf_counter()}")

    for i in range(0,n_xs):
        for j in range(0, n_xs):
            if j <= i : # matrix is symmetric
                val = calc_covariance(x[i], x[j], sigma_f, L)
                k_cap[i, j] = val
                k_cap[j, i] = val


    print(f"\n d {time.perf_counter()}")
    # print(k_cap)
    # print(k_cap.shape)

    k_cap = k_cap + (np.identity(n_xs) * sigma_n)
    print(f"\n e {time.perf_counter()}")

    return k_cap

def calc_K_batch(x, sigma_f, L, sigma_n):
    print("in calc_K")
    print(f"\n a {time.perf_counter()}")

    n_xs = len(x)
    print(f"\n b {time.perf_counter()}")
    k_cap = np.zeros((n_xs, n_xs))
    # print(f"k_cap here = {k_cap}")
    print(f"\n c {time.perf_counter()}")

    for i in range(0,n_xs):
        # matrix is symmetric, so less is calculated as time goes on
        val = calc_covariance_batch(x[i], x[i:], sigma_f, L)
        k_cap[i,i:] = val
        k_cap[i:,i] = val


    print(f"\n d {time.perf_counter()}")
    # print(k_cap)
    # print(k_cap.shape)

    k_cap = k_cap + (np.identity(n_xs) * sigma_n)
    print(f"\n e {time.perf_counter()}")

    return k_cap

def calc_covariance_batch(x, x_primes, sigma_f, L):
    # print("in calc_covariance")

    # # Check everything is same sizes # this slows optimisation so uncomment if
    # # there are errors to help diagnose bugs
    # if not np.isscalar(sigma_f):
    #     raise NameError(f"sigma_f must be a scalar, not an array: {sigma_f} {type(sigma_f)}")
    #
    # if np.isscalar(x):
    #     if not np.isscalar(L):
    #         raise NameError(f"x is scalar ({type(x)}) but L is {type(L)}")
    #
    # # todo check shape of x and x_prime
    # else:
    #     if np.isscalar(L):
    #         raise NameError(f"x is non-scalar ({type(x)}) but L is scalar ({type(L)})")
    #     if len(x) != len(L):
    #         raise NameError(f"Dimensions of x do not match number of Ls")

    # print(f" x is {x} x_primes is {x_primes}")

    x_diff = -x_primes + x
    # print(f"\n4 {time.perf_counter()}")
    # print(x_diff)

    x_diff_sqr = np.asarray(x_diff ** 2)  # asarray as can be both scalar or array
    # print(f"\n5 {time.perf_counter()}")
    x_sqr_l_sqr = np.asarray(np.nan_to_num(x_diff_sqr / (2 * (L ** 2))))
    # print(f"\n6 {time.perf_counter()}")
    x_sqr_l_sqr[x_sqr_l_sqr == np.inf] = 0  # remove infs
    # print(f"\n7 {time.perf_counter()}")

    sum_of_sqrs = np.sum(x_sqr_l_sqr,axis=1) # TODO this might break if things are 1d?
    # print(f"\n8 {time.perf_counter()}")

    exp_sum_sqr = np.exp(-sum_of_sqrs)
    # print(f"\n9 {time.perf_counter()}")

    k = exp_sum_sqr * (sigma_f ** 2)
    # print(f"\n10 {time.perf_counter()}")

    # # make sure returned stuff is correct types etc # again, uncomment to use
    # if np.isnan(k) or np.isinf(k):
    #     raise NameError("k contains not a number (has nan or inf)")
    #
    # if np.isscalar(k) == False:
    #     raise NameError("k is not scalar - must be a scalar")

    # print("end calc_covaraince")
    # print(k)

    return k

def calc_covariance(x, x_prime, sigma_f, L):
    # print("in calc_covariance")

    # # Check everything is same sizes # this slows optimisation so uncomment if
    # # there are errors to help diagnose bugs
    # if not np.isscalar(sigma_f):
    #     raise NameError(f"sigma_f must be a scalar, not an array: {sigma_f} {type(sigma_f)}")
    #
    # if np.isscalar(x):
    #     if not np.isscalar(L):
    #         raise NameError(f"x is scalar ({type(x)}) but L is {type(L)}")
    #
    # # todo check shape of x and x_prime
    # else:
    #     if np.isscalar(L):
    #         raise NameError(f"x is non-scalar ({type(x)}) but L is scalar ({type(L)})")
    #     if len(x) != len(L):
    #         raise NameError(f"Dimensions of x do not match number of Ls")



    x_diff = x - x_prime
    # print(f"\n4 {time.perf_counter()}")
    # print(x_diff)
    x_diff_sqr = np.asarray(x_diff ** 2)  # asarray as can be both scalar or array
    # print(f"\n5 {time.perf_counter()}")
    x_sqr_l_sqr = np.asarray(np.nan_to_num(x_diff_sqr / (2 * (L ** 2))))
    # print(f"\n6 {time.perf_counter()}")
    x_sqr_l_sqr[x_sqr_l_sqr == np.inf] = 0  # remove infs
    # print(f"\n7 {time.perf_counter()}")

    sum_of_sqrs = np.sum(x_sqr_l_sqr)
    # print(f"\n8 {time.perf_counter()}")

    exp_sum_sqr = np.exp(-sum_of_sqrs)
    # print(f"\n9 {time.perf_counter()}")

    k = exp_sum_sqr * (sigma_f ** 2)
    # print(f"\n10 {time.perf_counter()}")

    # # make sure returned stuff is correct types etc # again, uncomment to use
    # if np.isnan(k) or np.isinf(k):
    #     raise NameError("k contains not a number (has nan or inf)")
    #
    # if np.isscalar(k) == False:
    #     raise NameError("k is not scalar - must be a scalar")

    # print("end calc_covaraince")
    # print(k)

    return k

np.set_printoptions(precision=3,suppress=True)
x = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12],
    [13,14,15]
])
sigma_f = 1
ls = [2.7,5,1]
sigma_n_y = 0.5

print("### OLD CODE ###")
# original code
time_1 = time.perf_counter()
answer_1 = gp.calc_K(x, sigma_f, np.array(ls), sigma_n_y)
time_2 = time.perf_counter()

print(answer_1)
print(f"run time: {time_2 - time_1}")

print("### New CODE ###")
# hopefully faster but still functioning code
time_1 = time.perf_counter()
answer_2 = calc_K_faster(x, sigma_f, np.array(ls), sigma_n_y)
time_2 = time.perf_counter()

print(answer_2)
print(f"run time: {time_2 - time_1}")

print(f"both same: {(answer_2 == answer_1).all()}")

print("### Batch CODE ###")
# hopefully faster but still functioning code
time_1 = time.perf_counter()
answer_3 = calc_K_batch(x, sigma_f, np.array(ls), sigma_n_y)
time_2 = time.perf_counter()

print(answer_3)
print(f"run time: {time_2 - time_1}")


print(f"both same: {(answer_3 == answer_1).all()}")
