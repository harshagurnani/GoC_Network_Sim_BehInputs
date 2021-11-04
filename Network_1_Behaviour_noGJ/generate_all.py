import generate_beh_network_main as gb

for runid in range(3):
    for hom in [False]:
        for mf in [0, 0.2, 0.4, 0.6, 0.8]:
            for pf in [0, 0.2, 0.4, 0.6, 0.8]:
                gb.create_GoC_network(duration=20000, dt=0.025, runid=runid, mf=mf, pf=pf, hom=hom)
