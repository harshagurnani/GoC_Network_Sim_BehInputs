import generate_beh_network_main as gb

for runid in [0, 2, 4]:
    for ratio in [0.1, 0.3]:
        for mf in [0, 0.15, 0.3, 0.45]:
            for pf in [0, 0.15, 0.3, 0.45]:
                gb.create_GoC_network(
                    duration=20000,
                    dt=0.025,
                    runid=runid,
                    mf=mf,
                    pf=pf,
                    mf2=ratio * mf,
                    pf2=pf * ratio,
                    hom=False,
                )
