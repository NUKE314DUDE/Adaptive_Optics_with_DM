import numpy as np
import matplotlib.pyplot as plt
from DM_Control_Modules import smoothed_sawtooth

FILL = 0.95
CUT_LOW = 640; CUT_HIGH = None
SIG_FREQ = 10
amp_modulation = smoothed_sawtooth(FILL, CUT_LOW, CUT_HIGH, SIG_FREQ)
print(amp_modulation[0] == amp_modulation[-1], amp_modulation[0] - amp_modulation[-1])
t_stamp = np.linspace(0, 1/SIG_FREQ, len(amp_modulation))
plt.figure(figsize = (10, 6.18))
plt.plot(t_stamp, amp_modulation)
plt.xlabel("Time / s");plt.xticks(np.linspace(0, 1/SIG_FREQ, int(len(amp_modulation)/100)))
plt.ylabel("Modulation AMP")
plt.tight_layout()
plt.show()