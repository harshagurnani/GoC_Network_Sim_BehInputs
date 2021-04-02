import generate_beh_network_main as gb

for gje in [True, False]:
	for runid in [0,4,6]:
		for ratio in [0.2, 0.5, 1, 2]:
			for mf in [0.15, 0.3]:
				for pf in [0.15, 0.3]:
					gb.create_GoC_network( duration=20000, dt=0.025, runid=runid, mf=mf, pf=pf, mf2=ratio*mf, pf2=pf*ratio, gje=gje)
