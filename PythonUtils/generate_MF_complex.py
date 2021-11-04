import neuroml as nml
from pyneuroml import pynml

import sys
sys.path.append('../PythonUtils/')

import generate_inhom_spikearray as gip
import numpy as np

def create_spkarray_file( windowLim=[0, 400], duration=20, seedstart=100, nArr=200, name="MFON", scale=50, offset=7, minrate=2, write=True, behfile="D:\Work\OneDrive - University College London\models\dimensionality\Parameters\examplebeh.txt", wtlen=7, sparsef=0, wtMean=[], wtCov=[], rand_win=False, minWt=0 ):
    '''
        Create an nml file with a list of spike arrays, by using allBehFile,
        and other parameters to specify the time varying parameter rates.
        The k-th array will use the seed = seedstart+k.

        Using create_rate() in generate_inhom_spikearray:
            rate = (beh * weights) * scale + offset
            rate[rate<minrate] = minrate

        Inputs:
        - window    [List of 2 floats] = What window (in seconds) to crop behavioural traces in, to specify Rates
        - seedstart [INT] = specify number added to seed for each spike array
        - nArr      [INT] = Number of spike trains to create
        - name      [STR] = String added to id of each SpikeArray, and to NML file created
        - scale     [FLOAT] = Parameter passed to create_rate()
        - offset    [FLOAT] = Parameter passed to create_rate()
        - minrate   [FLOAT] = Parameter passed to create_rate()
        - write     [BOOL] = Write NML file?
    '''

    arr_flnm = 'SpikeArray_' + name + '_dur_' + format(int(duration),'03d') + '.nml'
    arrdoc = nml.NeuroMLDocument( id=name )

    bint = 0.1

    if not wtMean:
	    wtMean = np.ones(shape=(wtlen)) *0.5
    try:
	    if not wtCov.any():
	        wtCov = np.identity( wtlen )*0.03
    except:
        if not wtCov:
	        wtCov = np.identity( wtlen )*0.03
			
    print(wtMean, wtCov)
	
    allrates = {}
    allweights = {}
    for jj in range(nArr):
        np.random.seed(seedstart+jj)
	    
        if rand_win:
            start = np.random.random() * (windowLim[1]-windowLim[0]) + windowLim[0]
            window= [start, start+duration]
        else:
            window=[windowLim[0], windowLim[0]+duration]
		
        weights = np.random.multivariate_normal( wtMean, wtCov, 1).transpose()
        weights[np.random.random(size=(wtlen,1))<sparsef] = 0 #sparsify
        weights[weights<minWt] = 0
        time, rate = gip.create_rate( behfile, weights=weights, minT=window[0], maxT=window[1], scale=scale, offset=offset, minrate=minrate )
        inp = gip.create_spike_array( rate, duration, bint, name+'_{}'.format(jj) )
        allrates[jj] = rate
        allweights[jj] = weights
        arrdoc.spike_arrays.append( inp )


    if write:
        pynml.write_neuroml2_file( arrdoc, arr_flnm )
    
	return inp, allweights, allrates
