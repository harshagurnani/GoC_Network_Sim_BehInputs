# Generates file params_file.pkl, which enumerates parameter sets to be run in parallel
# Run as: python initialize_network_params.py

import numpy as np
import pickle as pkl

import network_utils as nu


def get_simulation_params(
    simid,
    nGoC=0,
    volume=[350, 350, 80],
    GoC_loc_type="Density",
    GoC_density=4607,
    GJ_dist_type="Boltzmann",
    GJ_wt_type="Szo16_oneGJ",
    protocol="TransientBurst",
    nBurst=4,
    Burst_type="Poisson",
    Burst_freq="100 Hz",
    Burst_delay="2000 ms",
    Burst_duration="500 ms",
    Burst_synapse="MF_GoC_SynMult.nml",
    Burst_conntype="random_sample",
    Burst_connprob=0,
    Burst_connGoC=4,
    nMFInput=15,
    MFInput_loc_type="Density",
    MF_density=0,
    MFInput_type="Poisson",
    MFInput_freq="5 Hz",
    MFInput_synapse="MF_GoC_Mult",
    MFInput_conntype="random_prob",
    MFInput_connprob=0.3,
    MFInput_connGoC=0,
):

    params = {}
    params["nGoC"], params["GoC_pos"] = nu.locate_GoC(nGoC, volume, GoC_loc_type, GoC_density, seed=simid)

    params["GJ_pairs"], params["GJ_wt"], params["GJ_loc"] = nu.GJ_conn(params["GoC_pos"], GJ_dist_type, GJ_wt_type, nDend=3, seed=simid)

    (params["nMF"], params["MF_pos"], params["MF_GoC_pairs"], params["MF_GoC_wt"],) = nu.MF_conn(
        nMFInput,
        MFInput_loc_type,
        volume,
        MF_density,
        params["GoC_pos"],
        MFInput_conntype,
        MFInput_connprob,
        MFInput_connGoC,
        seed=simid,
    )

    params["nBurst"] = nBurst
    params["Burst_GoC"] = nu.get_perturbed_GoC(params["nGoC"], Burst_conntype, Burst_connprob, Burst_connGoC, seed=simid)

    return params


if __name__ == "__main__":

    nSim = 10
    params_list = [get_simulation_params(simid) for simid in range(nSim)]

    file = open("params_file.pkl", "wb")
    pkl.dump(params_list, file)
    file.close()
