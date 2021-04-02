import generate_beh_network_main as gb

for runid in [0,4,6]:
	for ratio in [0.2, 0.4]:
		for mf in [0,  0.3,  .6]:
			for pf in [0, 0.3,  0.6]:
				gb.create_GoC_network( duration=20000, dt=0.025, runid=runid, mf=mf, pf=pf, mf2=ratio*mf, pf2=pf*ratio, hom=False)
