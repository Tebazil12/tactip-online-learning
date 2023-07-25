import tactip_toolkit_dobot.experiments.online_learning.offline_setup.data_processing as dp

a = [[-10], [ -8], [ -6], [ -4], [ -2], [  0], [  2], [  4], [  6], [  8], [ 10]]

# a = [[2]]

d = dp.add_line_mu(a, 4)

print(d)
