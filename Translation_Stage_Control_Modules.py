import numpy as np
from pipython import GCSDevice, pitools

translation_stage = GCSDevice('C-413.26A')
translation_stage.ConnectUSB(serialnum = '119024343')
pitools.startup(translation_stage, )