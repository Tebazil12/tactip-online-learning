import numpy as np
from tactip_toolkit_dobot.experiments.online_learning.contour_following_2d import (
    Experiment,
    make_meta,
    next_sensor_placement,
)

meta = make_meta()
ex = Experiment()

ex.edge_locations = [np.array([0,0]),np.array([0,10])]

orient, location = next_sensor_placement(ex,meta)

print(meta["STEP_LENGTH"])
print( str(location) )
