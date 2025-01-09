from Lib64.asdk import DM
import numpy
import time


def relax_mirror(dm, flat):
    t_start=time.perf_counter()
    while time.perf_counter()-t_start<4.0:
        amp=flat+numpy.exp(-(time.perf_counter()-t_start)/0.5)*numpy.cos((time.perf_counter()-t_start)/0.1*2*numpy.pi)
        dm.Send(numpy.ones(69)*amp)

flat=numpy.load("flat_voltages.npy")
# flat=numpy.zeros(69)
dm = DM( "BOL131" )
relax_mirror(dm,flat)
input()
