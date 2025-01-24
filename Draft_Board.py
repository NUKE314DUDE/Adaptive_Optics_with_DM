import os
os.add_dll_directory(os.getcwd())
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
os.chdir(current_directory)
import numpy as np
import matplotlib.pyplot as plt
import time
from Misc_Tools import frankot_chellappa, peaks
from Zernike_Polynomials_Modules import show_profile
from scipy import signal

X, Y, Z = peaks(200, 200,normalize=False)
grad_x, grad_y = np.gradient(Z)
show_profile(Z, X, Y)
# show_profile(grad_x, X, Y)
# show_profile(grad_y, X, Y)
re_Z = frankot_chellappa(grad_x, grad_y)
show_profile(re_Z, X, Y)
Z_ran = np.max(Z) - np.min(Z)
re_Z_ran = np.max(re_Z) - np.min(re_Z)
print((Z_ran - re_Z_ran)/Z_ran)
plt.show()