pyceres_location="../build/lib"
import sys
sys.path.append(pyceres_location)
import numpy as np

import PyCeres as ceres
# ceres.main_ceres()
target_v = np.array([1.0, 2.0, 3.0])
target_w = np.array([1.0, 0.0, 5.0])
ret = ceres.stick_optimization(target_v, target_w, 1, 0.1, 0.1)
print(target_v, "-->", ret[:3])
print(target_w, "-->", ret[3:])