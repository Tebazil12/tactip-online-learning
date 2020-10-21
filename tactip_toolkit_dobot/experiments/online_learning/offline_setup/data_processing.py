import numpy as np

def add_mus(disps, mu_limits=[-1, 1], line_ordering=None):
    """
    Disps must be a list, with each entry containing a np.array of
    displacement values (i.e. an entry per line)
    """
    # print(disps)

    n_disp_lines = len(disps)

    # [np.empty(disps[0].shape)] * len(disps)
    x = np.empty((n_disp_lines, disps[0].shape[0], 2))

    mu_for_line = np.linspace(mu_limits[0], mu_limits[1], n_disp_lines)
    # print(mu_for_line)

    for i, disp in enumerate(disps):
        if line_ordering is not None:
            if len(line_ordering) != n_disp_lines:
                raise NameError("line_ordering has different length to disps")
            # print(line_ordering[i])
            mus = np.full(disp.shape, mu_for_line[line_ordering[i]])
        else:
            mus = np.full(disp.shape, mu_for_line[i])
        # print(mus)
        # print(type(mus))

        an_x = np.vstack((disp, mus))
        # print(an_x)

        x[i] = an_x.T  # todo is this ok, or should np.copy be used for assignment?

    # print(x)
    # print(x.shape)
    return x