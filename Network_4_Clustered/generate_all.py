import generate_beh_network_main as gb
from numpy import random as rn

for runid in [1,3,5]:
	for oo in range(3):
		order = rn.permutation(4)
		for gje in [True, False]:
			for mf in [0.2, 0.4, 0.6]:
				for pf in [0, 0.2, 0.4]:
					gb.create_GoC_network( duration=40000, dt=0.025, runid=runid, mf=mf, pf=pf, order=order, gje=gje)
