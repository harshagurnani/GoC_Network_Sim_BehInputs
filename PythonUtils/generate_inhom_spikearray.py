import neuroml as nml
import numpy as np
import math

import sys

sys.path.append("../PythonUtils/")


def create_spike_array(rate, maxT, dt, name):
    """
    Create inhomogeneous Poisson process
    Based on: https://hpaulkeeler.com/simulating-an-inhomogeneous-poisson-point-process/

    Inputs:
    - rate (array of rate sampled at dt)
    - maxT [FLOAT] = maximum spiketime (sec)
    - dt [FLOAT] = binning used to generate points (sec) (~0.1s is fine, smaller for faster varying rates)
    - name [STR] = Name of spike array
    """

    x = np.arange(0, maxT, dt)
    m = getrate(rate, x, dt)  # [0:len(x)]

    m2 = max(m)[0]  # reweighted
    # generate uniform random variable
    u = np.random.random(size=(np.int(np.ceil(1.5 * maxT * m2)), 1))
    y = np.cumsum(-1 / m2 * np.log(u))  # transform to homogeneous Poisson
    y = y[y <= maxT]

    # filter some points with probability = rate(t)/maxrate
    prob = np.reshape(np.random.random(size=y.shape), [len(y), 1])
    l2 = np.reshape(getrate(rate, y, dt) / m2, [len(y), 1])
    keep = np.where(prob < l2)
    y = y[keep[0]]

    input = nml.SpikeArray(id=name)
    ctr = 0
    for spk in y:
        input.spikes.append(nml.Spike(id=ctr, time="{} ms".format(1e3 * spk)))

    return input


def create_rate(allBehFile, weights, minT, maxT, scale=10, offset=5, minrate=2):
    allbeh = np.genfromtxt(allBehFile, delimiter=",")

    # Crop timeseries at assigned points
    allbeh = allbeh[(allbeh[:, 0] >= minT * 1e3) & (allbeh[:, 0] < maxT * 1e3), :]
    time = allbeh[:, 0]
    # convert behaviour to firing rate
    rate = np.matmul(allbeh[:, 1:], weights) * scale + offset
    rate[rate < minrate] = minrate

    return time, rate


def sinrate(t):
    """
    Function that returns firing rate (Hz) by a predefined sin function, using input t
    """
    return 50 * np.sin(t / 10)


def getrate(rate, x, dt):
    r = [rate[np.int(np.floor(x[jj] / dt))] for jj in range(len(x))]
    return np.array(r)
