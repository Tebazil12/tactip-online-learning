import numpy as np


def max_log_like(hyper_pars, data):
    # print(hyper_pars)
    [sigma_f, ls] = hyper_pars
    [x, y, sigma_n] = data

    # k_cap = calc_K(x, sigma_f, L, sigma_n)
    #
    # # with np.printoptions(precision=3, suppress=True):
    # # print(k_cap)
    # # print("########################################")
    #
    # r_cap = np.linalg.cholesky(k_cap)
    #
    # sign, logdet_K = np.linalg.slogdet(r_cap)
    #
    # alpha = np.linalg.solve(r_cap, np.linalg.solve(r_cap.conj().T, y))
    #
    # # todo check its doing matrix mult. not element wise mult
    # mll = (
    #     (0.5 * (y.conj().T @ alpha))
    #     + 0.5 * logdet_K
    #     + 0.5 * k_cap.shape[0] * np.log(2 * np.pi)
    # )
    #
    # # print(mll)
    # # raise(stop)
    #
    # # print(".", end='')
    #
    # return mll

    np.set_printoptions(suppress=True)
    # print(f"max log lik x = {np.round(x,1)} and y={np.round(y,1)}")
    # print(y.shape)
    d_cap, n_cap = y.shape

    s_cap = (1 / d_cap) * (y @ y.conj().T)

    # if not np.isscalar(s_cap):
    #     raise NameError("s_cap is not scalar!")

    # ls = np.array([l_disp, l_mu])

    k_cap = calc_K(x, sigma_f, np.array(ls), sigma_n)

    r_cap = np.linalg.cholesky(k_cap)
    sign, logdet_K = np.linalg.slogdet(r_cap)

    part_1 = -(d_cap * n_cap) * 0.5 * np.log(2 * np.pi)
    part_2 = -d_cap * 0.5 * logdet_K

    # print("Here")
    # print(d_cap)
    # print(s_cap)
    # print(np.trace(np.linalg.inv(k_cap)))
    part_3 = -d_cap * 0.5 * np.trace(np.linalg.inv(k_cap) @ s_cap)

    neg_val = part_1 + part_2 + part_3
    # print(f"max log like is: {-neg_val}")
    print(".",end='')
    # print(-neg_val)
    return -neg_val  # because trying to find max with a min search


def calc_K(x, sigma_f, L, sigma_n):
    np.set_printoptions(suppress=True)
    n_xs = len(x)

    k_cap = np.empty([n_xs, n_xs])

    for i in range(0,n_xs):
        # matrix is symmetric, so less is calculated as time goes on
        # print(f"calcing x[i]={x[i]} and x[i:] = {x[i:]}")
        val = calc_covariance(x[i], x[i:], sigma_f, L)
        # val = calc_covariance(x[i], x, sigma_f, L)
        # print(f"val = {val}")
        k_cap[i,i:] = val
        k_cap[i:,i] = val
    # for i in range(0, n_xs):
    #     for j in range(0, n_xs):
    #         k_cap[i, j] = calc_covariance(x[i], x[j], sigma_f, L)

    # print(k_cap)
    # print(k_cap.shape)

    k_cap = k_cap + (np.identity(n_xs) * sigma_n)
    # print(f"kcap = {k_cap}")

    return k_cap


def cal_K_star(x, x_star, sigma_f, L, sigma_n):
    # print(len(x))
    # print(x)
    # print(x_star)
    # print(sigma_f)
    # print(L)
    # print(x_star.shape)

    k_cap_star = np.empty(x.shape)
    # print(len(x.shape))
    if len(x.shape) == 1:
        x = np.array([x]).T
        if len(x.shape) == 1:
            raise NameError("wtf")
    # for i in range(0, len(k_cap_star)):
    #     k_cap_star[i] = calc_covariance(x_star, x[i], sigma_f, L)

    k_cap_star = calc_covariance(x_star, x, sigma_f, L)

    if np.isnan(k_cap_star).any():
        raise NameError("Entry in k_cap_star is nan")

    # with np.printoptions(precision=3, suppress=True):
    #     print(k_cap_star)
    return k_cap_star


def calc_covariance(x, x_primes, sigma_f, L):
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

    # print(f"type of x and xprime os {type(x)} {type(x_primes)}")
    if type(x) is not np.ndarray:
        # print(f"warning, x {x} was not an array")
        x = np.array(x)
    if len(x.shape) != 2:
        # print(f"warning, x {x} was not 2d (first)")
        x = np.array([x])

    if len(x.shape) != 2:
        # print(f"warning, x {x} was not 2d (second)")
        x = np.array([x])

    # print(f"x is now {x}")

    if type(x_primes) is not np.ndarray:
        # print("warning, xprimes was not an array")
        x_primes = np.array(x_primes)
    if len(x_primes.shape) != 2:
        # print("warning, xprimes was not 2d (first)")
        x_primes = np.array([x_primes]).T
    if len(x_primes.shape) != 2:
        x_primes = np.array([x_primes]).T
        # print("warning, xprimes was not 2d (second)")

    # print(f"x is {x.shape} and xprime is {x_primes.shape}")

    x_diff = -x_primes + x
    # print(x_diff)
    x_diff_sqr = np.asarray(x_diff ** 2)  # asarray as can be both scalar or array

    x_sqr_l_sqr = np.asarray(np.nan_to_num(x_diff_sqr / (2 * (L ** 2))))

    x_sqr_l_sqr[x_sqr_l_sqr == np.inf] = 0  # remove infs

    if np.shape(np.shape(x_sqr_l_sqr))[0] == 2:
        sum_of_sqrs = np.sum(x_sqr_l_sqr,axis=1)
    else:
        sum_of_sqrs = np.sum(x_sqr_l_sqr)

    # print(f"sumofsqrs is type {type(sum_of_sqrs)}{type(sum_of_sqrs[0])}, shape {np.shape(sum_of_sqrs)} = {sum_of_sqrs}")
    # print(np)
    exp_sum_sqr = np.exp(-sum_of_sqrs)

    k = exp_sum_sqr * (sigma_f ** 2)


    # # make sure returned stuff is correct types etc # again, uncomment to use
    # if np.isnan(k) or np.isinf(k):
    #     raise NameError("k contains not a number (has nan or inf)")
    #
    # if np.isscalar(k) == False:
    #     raise NameError("k is not scalar - must be a scalar")

    # print("end calc_covaraince")
    # print(k)

    return k


def interpolate(x, y, sigma_f, L, sigma_n, x_limits=None, x_step=0.1, mu=None, height=None):
    """
    Given example data, x and y, and (optimised) hyperparameters, use a gp
    to interpolate y at a range of x's between the x_limits (both inclusive).
    If x_limits are not set, use the max and min of x as limits
    """
    y = np.array([y])  # make 2d, so that transpose works...

    k_cap = calc_K(x, sigma_f, L, sigma_n)
    # k_cap = np.array([k_cap])  # so that transpose works...


    if x_limits is None:
        x_stars = np.arange(np.amin(x), np.amax(x) + x_step, x_step)
    else:
        x_stars = np.arange(x_limits[0], x_limits[1] + x_step, x_step)
    len_x_stars = len(x_stars)
    if height is not None and mu is not None:
        x_stars = np.array([x_stars, np.ones(len_x_stars)*height, np.ones(len_x_stars)*mu]).T



    # x_stars = np.around(x_stars, 4)  # get rid of tiny errors

    y_stars = np.empty(x_stars.shape)

    for i, x_star in enumerate(x_stars):
        # print(i, x_star)

        # print(x.shape)
        # print(x_star.shape)
        # print(f"x_star is {x_star}")
        # dfg

        k_cap_star = cal_K_star(x, x_star, sigma_f, L, sigma_n)
        k_cap_star = np.array([k_cap_star])
        # print("kcapstar")
        # print(k_cap_star)


        # print("k_cap_star")
        # print(k_cap_star.shape)
        # print(k_cap_star)

        # print("inv_k")
        inv_K = np.linalg.inv(k_cap)
        # print(inv_K)
        # print(inv_K.shape)

        # print("y and y apostrophe")
        # print(y.shape)
        y_con_t = y.conj().T
        # print(y_con_t)

        # print(y_con_t.shape)

        # print("K* x inv(K) x y_appostrophe ")
        ks_by_yt = k_cap_star @ inv_K @ y_con_t  # the @ means matrix mult.
        # print(ks_by_yt.shape)
        # print(ks_by_yt)


        y_stars[i] = ks_by_yt
        # print("y_starsi")
        # print(y_stars[i])

        # raise(stop)

        # k_star_star = calc_covariance(x_star, x_star, sigma_f, L)
    # with np.printoptions(precision=3, suppress=True):
    #     print(y_stars)
    return x_stars, y_stars
