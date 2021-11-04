# Generates file params_file.pkl, which enumerates parameter sets to be run in parallel
# Run as: python initialize_network_params.py

import numpy as np
import pickle as pkl


def get_channel_params(
    simid,
    ra=0.1,
    defaults={
        "leak_cond": 0.021,
        "na_cond": 48.0,
        "nap_cond": 0.19,
        "nar_cond": 1.7,
        "ka_cond": 8.0,
        "sk2_cond": 38.0,
        "kv_cond": 32.0,
        "km_cond": 1.0,
        "bk_cond": 3.0,
        "hcn1f_cond": 0.05,
        "hcn1s_cond": 0.05,
        "hcn2f_cond": 0.08,
        "hcn2s_cond": 0.08,
        "cahva_cond": 0.46,
        "calva_cond": 0.25,
    },
):

    params = {}
    np.random.seed(simid)
    params["ra"] = "{} kohm_cm".format(ra * (1 + 0.2 * (np.random.random(1)[0] - 0.5)))
    for cond, value in defaults.items():
        params[cond] = "{} mS_per_cm2".format(value * (1 + 0.4 * (np.random.random(1)[0] - 0.5)))
    return params


if __name__ == "__main__":

    ra = 0.1
    defaults = {
        "leak_cond": 0.021,
        "na_cond": 48.0,
        "nap_cond": 0.19,
        "nar_cond": 1.7,
        "ka_cond": 8.0,
        "sk2_cond": 38.0,
        "kv_cond": 32.0,
        "km_cond": 1.0,
        "bk_cond": 3.0,
        "hcn1f_cond": 0.05,
        "hcn1s_cond": 0.05,
        "hcn2f_cond": 0.08,
        "hcn2s_cond": 0.08,
        "cahva_cond": 0.46,
        "calva_cond": 0.25,
    }

    nSim = 1000
    params_list = [get_channel_params(simid, ra, defaults) for simid in range(nSim)]

    file = open("cellparams_file.pkl", "wb")
    pkl.dump(params_list, file)
    file.close()
