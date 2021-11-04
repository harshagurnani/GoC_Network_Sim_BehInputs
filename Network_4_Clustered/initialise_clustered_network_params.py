# Generates file binetwork_params.pkl -> parameters for full model og Golgi network with ON/OFF MF and PF inputs
# Run as: python initialize_bimodal_network_params.py
# Or load module and call get_simulation_params() withh different arguments

import numpy as np
from numpy.core import multiarray
import pickle as pkl

import sys

sys.path.append("../PythonUtils")

import network_utils as nu


def get_simulation_params(
    simid,
    sim_durn=20000,
    nGoC=0,
    nGoC_pop=5,
    densityParams="../Parameters/useParams_FI_14_25.pkl",
    volume=[500, 500, 100],
    GoC_density=4607,
    GJ_dist_type="Boltzmann",
    GJ_wt_type="Szo16_oneGJ",
    nGJ_dend=3,
    GoC_wtscale=1.0,
    GoC_probscale=1.0,
    stepInputs=["MF_Step", "PF_Step"],
    bgInputs=["MF_bg", "PF_bg"],
    behInputs=["MFON", "PFON", "MFOFF", "PFOFF", "MFRand", "PFRand"],
    clusInputs=[
        "LocoMF",
        "LocoPF",
        "WMIAmpMF",
        "WMIAmpPF",
        "PupilMF",
        "PupilPF",
        "WSPMF",
        "WSPPF",
    ],
    clusType={
        "LocoMF": "BehMF",
        "LocoPF": "BehPF",
        "WMIAmpMF": "BehMF",
        "WMIAmpPF": "BehPF",
        "PupilMF": "BehMF",
        "PupilPF": "BehPF",
        "WSPMF": "BehMF",
        "WSPPF": "BehPF",
    },
    input_nFiles={
        "MF_Step": 300,
        "MF_bg": 60,
        "PF_Step": 300,
        "PF_bg": 60,
        "MFON": 200,
        "PFON": 200,
        "MFOFF": 200,
        "PFOFF": 200,
        "MFRand": 200,
        "PFRand": 300,
        "BehMF": 100,
        "BehPF": 100,
    },
    nInputs_max={
        "MF_Step": 60,
        "MF_bg": 60,
        "PF_Step": 300,
        "PF_bg": 300,
        "MFON": 60,
        "PFON": 300,
        "MFOFF": 60,
        "PFOFF": 300,
        "MFRand": 60,
        "PFRand": 300,
        "BehMF": 15,
        "BehPF": 30,
    },
    nInputs_frac={
        "MF_Step": 0.55,
        "MF_bg": 0.10,
        "PF_Step": 0.45,
        "PF_bg": 0.35,
        "MFON": 0.5,
        "PFON": 0.5,
        "MFOFF": 0.5,
        "PFOFF": 0.5,
        "MFRand": 0.5,
        "PFRand": 0.5,
        "BehMF": 0.5,
        "BehPF": 0.5,
    },
    Input_density={
        "MF_Step": 0,
        "MF_bg": 0,
        "PF_Step": 0,
        "PF_bg": 0,
        "MFON": 0,
        "PFON": 0,
        "MFOFF": 0,
        "PFOFF": 0,
        "MFRand": 0,
        "PFRand": 0,
        "BehMF": 0,
        "BehPF": 0,
    },
    Input_dend={
        "MF_Step": 3,
        "MF_bg": 3,
        "PF_Step": 3,
        "PF_bg": 3,
        "MFON": 3,
        "PFON": 3,
        "MFOFF": 3,
        "PFOFF": 3,
        "MFRand": 3,
        "PFRand": 3,
        "BehMF": 3,
        "BehPF": 3,
    },
    Input_nRosette={
        "MF_Step": 0,
        "MF_bg": 0,
        "PF_Step": 0,
        "PF_bg": 0,
        "MFON": 0,
        "PFON": 0,
        "MFOFF": 0,
        "PFOFF": 0,
        "MFRand": 0,
        "PFRand": 0,
        "BehMF": 3,
        "BehPF": 3,
    },
    Input_loc={
        "MF_Step": "random",
        "MF_bg": "random",
        "PF_Step": "random",
        "PF_bg": "random",
        "MFON": "random",
        "PFON": "random",
        "MFOFF": "random",
        "PFOFF": "random",
        "MFRand": "random",
        "PFRand": "random",
        "BehMF": "random",
        "BehPF": "random",
    },
    Input_type={
        "MF_Step": "MF_step",
        "MF_bg": "poisson",
        "PF_Step": "PF_step",
        "PF_bg": "poisson",
        "MFON": "MFON",
        "PFON": "PFON",
        "MFOFF": "MFOFF",
        "PFOFF": "PFOFF",
        "MFRand": "MFRand",
        "PFRand": "PFRand",
    },
    Input_id={
        "MF_Step": 1,
        "MF_bg": 0,
        "PF_Step": 1,
        "PF_bg": 0,
        "MFON": 2,
        "PFON": 3,
        "MFOFF": 4,
        "PFOFF": 5,
        "MFRand": 6,
        "PFRand": 7,
    },
    Input_syn={
        "MF_Step": ["../Mechanisms/MF_GoC_Syn.nml", "ExpThreeSynapse"],
        "MF_bg": ["../Mechanisms/MF_GoC_Syn.nml", "ExpThreeSynapse"],
        "PF_Step": ["../Mechanisms/PF_GoC_Syn.nml", "ExpTwoSynapse"],
        "PF_bg": ["../Mechanisms/PF_GoC_Syn.nml", "ExpTwoSynapse"],
        "MFON": ["../Mechanisms/MF_GoC_Syn.nml", "ExpThreeSynapse"],
        "PFON": ["../Mechanisms/PF_GoC_Syn.nml", "ExpTwoSynapse"],
        "MFOFF": ["../Mechanisms/MF_GoC_Syn.nml", "ExpThreeSynapse"],
        "PFOFF": ["../Mechanisms/PF_GoC_Syn.nml", "ExpTwoSynapse"],
        "MFRand": ["../Mechanisms/MF_GoC_Syn.nml", "ExpThreeSynapse"],
        "PFRand": ["../Mechanisms/PF_GoC_Syn.nml", "ExpTwoSynapse"],
        "BehMF": ["../Mechanisms/MF_GoC_Syn.nml", "ExpThreeSynapse"],
        "BehPF": ["../Mechanisms/PF_GoC_Syn.nml", "ExpTwoSynapse"],
    },
    Input_conn={
        "MF_Step": "random_prob",
        "MF_bg": "random_prob",
        "PF_Step": "random_prob",
        "PF_bg": "random_prob",
        "MFON": "random_prob",
        "PFON": "random_prob",
        "MFOFF": "random_prob",
        "PFOFF": "random_prob",
        "MFRand": "random_prob",
        "PFRand": "random_prob",
        "BehMF": "random_prob",
        "BehPF": "random_prob",
    },
    Input_prob={
        "MF_Step": 0.3,
        "MF_bg": 0.3,
        "PF_Step": 0.3,
        "PF_bg": 0.8,
        "MFON": 0.3,
        "PFON": 0.3,
        "MFOFF": 0.3,
        "PFOFF": 0.3,
        "MFRand": 0.3,
        "PFRand": 0.3,
        "BehMF": 0.3,
        "BehPF": 0.3,
    },
    Input_rate={"MF_bg": [5], "PF_bg": [2]},
    Input_nGoC={
        "MF_Step": 0,
        "MF_bg": 0,
        "PF_Step": 0,
        "PF_bg": 0,
        "MFON": 0,
        "PFON": 0,
        "MFOFF": 0,
        "PFOFF": 0,
        "MFRand": 0,
        "PFRand": 0,
        "BehMF": 0,
        "BehPF": 0,
    },
    Input_wt={
        "MF_Step": 1,
        "MF_bg": 1,
        "PF_Step": 1,
        "PF_bg": 1,
        "MFON": 1,
        "PFON": 1,
        "MFOFF": 1,
        "PFOFF": 1,
        "MFRand": 1,
        "PFRand": 1,
        "BehMF": 1,
        "BehPF": 1,
    },
    Input_maxD={
        "MF_Step": [300],
        "MF_bg": [300],
        "PF_Step": [100, 2000, 300],
        "PF_bg": [100, 2000, 300],
        "MFON": [300],
        "PFON": [100, 2000, 300],
        "MFOFF": [300],
        "PFOFF": [100, 2000, 300],
        "MFRand": [300],
        "PFRand": [100, 2000, 300],
        "BehMF": [300, 150, 300],
        "BehPF": [100, 2000, 300],
    },
    Input_cellloc={
        "MF_Step": "soma",
        "MF_bg": "soma",
        "PF_Step": "dend",
        "PF_bg": "dend",
        "MFON": "soma",
        "PFON": "dend",
        "MFOFF": "soma",
        "PFOFF": "dend",
        "MFRand": "soma",
        "PFRand": "dend",
        "BehMF": "soma",
        "BehPF": "dend",
    },
    Input_flnm={
        "MFON": "../Mechanisms/SpikeArray_MFON_dur_060_080.nml",
        "PFON": "../Mechanisms/SpikeArray_MFON_dur_060_080.nml",
        "MFOFF": "../Mechanisms/SpikeArray_MFOFF_dur_060_080.nml",
        "PFOFF": "../Mechanisms/SpikeArray_MFON_dur_060_080.nml",
        "MFRand": "../Mechanisms/SpikeArray_RandomWin_MF_dur_020.nml",
        "PFRand": "../Mechanisms/SpikeArray_RandomWin_PF_dur_020.nml",
        "LocoMF": "../Mechanisms/SpikeArray_Loco_dur_040.nml",
        "LocoPF": "../Mechanisms/SpikeArray_Loco_PF_dur_040.nml",
        "WMIAmpMF": "../Mechanisms/SpikeArray_WMIAmp_dur_040.nml",
        "WMIAmpPF": "../Mechanisms/SpikeArray_WMIAmp_PF_dur_040.nml",
        "PupilMF": "../Mechanisms/SpikeArray_Pupil_dur_040.nml",
        "PupilPF": "../Mechanisms/SpikeArray_Pupil_PF_dur_040.nml",
        "WSPMF": "../Mechanisms/SpikeArray_WSP_dur_040.nml",
        "WSPPF": "../Mechanisms/SpikeArray_WSP_PF_dur_040.nml",
    },
    connect_goc={
        "MF_Step": False,
        "MF_bg": False,
        "PF_Step": False,
        "PF_bg": False,
        "MFON": False,
        "PFON": False,
        "MFOFF": False,
        "PFOFF": False,
        "MFRand": False,
        "PFRand": False,
        "BehMF": False,
        "BehPF": False,
    },
    clusVolume={
        "LocoMF": [500, 125, 100],
        "LocoPF": [500, 125, 100],
        "WMIAmpMF": [500, 125, 100],
        "WMIAmpPF": [500, 125, 100],
        "PupilMF": [500, 125, 100],
        "PupilPF": [500, 125, 100],
        "WSPMF": [500, 125, 100],
        "WSPPF": [500, 125, 100],
    },
    clusSpatialOffset={
        "LocoMF": [0, 0, 0],
        "LocoPF": [0, 0, 0],
        "WMIAmpMF": [0, 130, 0],
        "WMIAmpPF": [0, 130, 0],
        "PupilMF": [0, 260, 0],
        "PupilPF": [0, 260, 0],
        "WSPMF": [0, 390, 0],
        "WSPPF": [0, 390, 0],
    },
):

    params = {}

    # 1. Golgi Population - Type, Location and Electrical Coupling

    params["nGoC"], params["GoC_pos"] = nu.locate_GoC(nGoC, volume, GoC_density, seed=simid)

    params["nPop"] = nGoC_pop
    params["GoC_ParamID"], params["nGoC"] = nu.get_hetero_GoC_id(params["nGoC"], nGoC_pop, densityParams, seed=simid)
    popsize = params["nGoC"] / params["nPop"]
    params["nGoC_per_pop"] = popsize
    params["nGoC"], params["GoC_pos"] = nu.locate_GoC(params["nGoC"], volume, GoC_density, seed=simid)

    print("Created parameters for {} GoC".format(params["nGoC"]))
    usedID = np.unique(np.asarray(params["GoC_ParamID"]))
    # Get connectivity and sort into populations:
    params["econn_pop"] = [[{} for post in range(nGoC_pop - pre)] for pre in range(nGoC_pop)]
    gj, wt, loc = nu.GJ_conn(
        params["GoC_pos"],
        GJ_dist_type,
        GJ_wt_type,
        nDend=nGJ_dend,
        wt_k=GoC_wtscale,
        prob_k=GoC_probscale,
        seed=simid,
    )

    for pre in range(nGoC_pop):
        for post in range(pre, nGoC_pop):
            pairid = [x for x in range(gj.shape[0]) if ((np.floor_divide(gj[x, 1], popsize) == post) & (np.floor_divide(gj[x, 0], popsize) == pre))]
            params["econn_pop"][pre][post - pre] = {
                "GJ_pairs": np.mod(gj[pairid, :], popsize),
                "GJ_wt": wt[pairid],
                "GJ_loc": loc[pairid, :],
            }

    params["Inputs"] = {"types": []}

    ctr = 1
    np.random.seed(simid + 1000)
    for input in bgInputs:
        Inp = {
            "type": Input_type[input],
            "rate": Input_rate[input],
            "syn_type": Input_syn[input],
            "syn_loc": Input_cellloc[input],
        }
        if connect_goc[input]:
            choose_goc = np.random.randint(params["nGoC"])
            (Inp["nInp"], Inp["pos"], Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"],) = nu.connect_inputs(
                maxn=nInputs_max[input],
                frac=nInputs_frac[input],
                density=Input_density[input],
                volume=volume,
                mult=Input_nRosette[input],
                connType=Input_conn[input],
                connProb=Input_prob[input],
                connGoC=Input_nGoC[input],
                connWeight=Input_wt[input],
                connDist=Input_maxD[input],
                GoC_pos=np.reshape(params["GoC_pos"][choose_goc, :], (1, 3)),
                cell_loc=Input_cellloc[input],
                seed=simid,
            )
            if len(Inp["conn_pairs"]) > 1:
                for jj in range(len(Inp["conn_pairs"][1])):
                    Inp["conn_pairs"][1][jj] = choose_goc
        else:
            (Inp["nInp"], Inp["pos"], Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"],) = nu.connect_inputs(
                maxn=nInputs_max[input],
                frac=nInputs_frac[input],
                density=Input_density[input],
                volume=volume,
                mult=Input_nRosette[input],
                connType=Input_conn[input],
                connProb=Input_prob[input],
                connGoC=Input_nGoC[input],
                connWeight=Input_wt[input],
                connDist=Input_maxD[input],
                GoC_pos=params["GoC_pos"],
                cell_loc=Input_cellloc[input],
                nDend=Input_dend[input],
                seed=simid * ctr,
            )
        tmp = {"conn_pairs": [], "conn_wt": [], "conn_loc": []}
        if Inp["nInp"] > 0:
            np.random.seed(simid * Inp["nInp"] * ctr)
            Inp["sample"] = np.random.permutation(input_nFiles[input])[0 : Inp["nInp"]]
            for pid in usedID:
                pairid = [x for x in range(Inp["conn_pairs"].shape[1]) if params["GoC_ParamID"][Inp["conn_pairs"][1, x]] == pid]
                tmp["conn_pairs"].append(Inp["conn_pairs"][:, pairid])
                tmp["conn_wt"].append(Inp["conn_wt"][pairid])
                tmp["conn_loc"].append(Inp["conn_loc"][:, pairid])
            for key in ["conn_pairs", "conn_wt", "conn_loc"]:
                Inp[key] = tmp[key]
            for jj in range(nGoC_pop):
                Inp["conn_pairs"][jj][1, :] = np.mod(Inp["conn_pairs"][jj][1, :], popsize)
            params["Inputs"]["types"].append(input)

        params["Inputs"][input] = Inp
        ctr += 1

    for input in stepInputs:

        for id in range(5):
            Inp = {
                "type": Input_type[input],
                "syn_type": Input_syn[input],
                "syn_loc": Input_cellloc[input],
                "id": id + 1,
            }
            if connect_goc[input]:
                choose_goc = np.random.randint(params["nGoC"])
                (Inp["nInp"], Inp["pos"], Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"],) = nu.connect_inputs(
                    maxn=nInputs_max[input],
                    frac=nInputs_frac[input],
                    density=Input_density[input],
                    volume=volume,
                    mult=Input_nRosette[input],
                    connType=Input_conn[input],
                    connProb=Input_prob[input],
                    connGoC=Input_nGoC[input],
                    connWeight=Input_wt[input],
                    connDist=Input_maxD[input],
                    GoC_pos=np.reshape(params["GoC_pos"][choose_goc, :], (1, 3)),
                    cell_loc=Input_cellloc[input],
                    seed=simid,
                )
                if len(Inp["conn_pairs"]) > 1:
                    for jj in range(len(Inp["conn_pairs"][1])):
                        Inp["conn_pairs"][1][jj] = choose_goc
            else:
                (Inp["nInp"], Inp["pos"], Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"],) = nu.connect_inputs(
                    maxn=nInputs_max[input],
                    frac=nInputs_frac[input],
                    density=Input_density[input],
                    volume=volume,
                    mult=Input_nRosette[input],
                    connType=Input_conn[input],
                    connProb=Input_prob[input],
                    connGoC=Input_nGoC[input],
                    connWeight=Input_wt[input],
                    connDist=Input_maxD[input],
                    GoC_pos=params["GoC_pos"],
                    cell_loc=Input_cellloc[input],
                    nDend=Input_dend[input],
                    seed=simid * ctr,
                )
            tmp = {"conn_pairs": [], "conn_wt": [], "conn_loc": []}
            if Inp["nInp"] > 0:
                np.random.seed(simid * Inp["nInp"] * ctr)
                Inp["sample"] = np.random.permutation(input_nFiles[input])[0 : Inp["nInp"]]
                for pid in usedID:
                    pairid = [x for x in range(Inp["conn_pairs"].shape[1]) if params["GoC_ParamID"][Inp["conn_pairs"][1, x]] == pid]
                    tmp["conn_pairs"].append(Inp["conn_pairs"][:, pairid])
                    tmp["conn_wt"].append(Inp["conn_wt"][pairid])
                    tmp["conn_loc"].append(Inp["conn_loc"][:, pairid])
                for key in ["conn_pairs", "conn_wt", "conn_loc"]:
                    Inp[key] = tmp[key]
                for jj in range(nGoC_pop):
                    Inp["conn_pairs"][jj][1, :] = np.mod(Inp["conn_pairs"][jj][1, :], popsize)
                params["Inputs"]["types"].append(input + format(id + 1))

            print(input + format(id + 1))
            params["Inputs"][input + format(id + 1)] = Inp
            ctr += 1

    for input in behInputs:
        Inp = {
            "type": Input_type[input],
            "syn_type": Input_syn[input],
            "syn_loc": Input_cellloc[input],
            "id": Input_id[input],
        }
        Inp["nInp"] = int(nInputs_max[input] * nInputs_frac[input])
        Inp["nInp"], Inp["pos"] = nu.locate_GoC(nGoC=Inp["nInp"], volume=volume, seed=simid + Input_id[input])
        Inp["filename"] = Input_flnm[input]

        tmp = {"conn_pairs": [], "conn_wt": [], "conn_loc": []}
        if Inp["nInp"] > 0:
            np.random.seed(simid * Inp["nInp"] * ctr + Input_id[input])
            Inp["sample"] = np.random.permutation(input_nFiles[input])[0 : Inp["nInp"]]  # choose wh

            (Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"],) = nu.connect_inputs_known(
                nInp=Inp["nInp"],
                Inp_pos=Inp["pos"],
                mult=Input_nRosette[input],
                connType=Input_conn[input],
                connProb=Input_prob[input],
                connGoC=Input_nGoC[input],
                connWeight=Input_wt[input],
                connDist=Input_maxD[input],
                GoC_pos=params["GoC_pos"],
                cell_loc=Input_cellloc[input],
                nDend=Input_dend[input],
                seed=simid * ctr + Input_id[input],
            )
            for pid in usedID:
                pairid = [x for x in range(Inp["conn_pairs"].shape[1]) if params["GoC_ParamID"][Inp["conn_pairs"][1, x]] == pid]
                tmp["conn_pairs"].append(Inp["conn_pairs"][:, pairid])
                tmp["conn_wt"].append(Inp["conn_wt"][pairid])
                tmp["conn_loc"].append(Inp["conn_loc"][:, pairid])
            for key in ["conn_pairs", "conn_wt", "conn_loc"]:
                Inp[key] = tmp[key]
            for jj in range(nGoC_pop):
                Inp["conn_pairs"][jj][1, :] = np.mod(Inp["conn_pairs"][jj][1, :], popsize)
            params["Inputs"]["types"].append(input)

        params["Inputs"][input] = Inp
        ctr += 1

    idctr = 7
    for input in clusInputs:
        idctr += 1
        spktype = clusType[input]
        Inp = {
            "type": spktype,
            "syn_type": Input_syn[spktype],
            "syn_loc": Input_cellloc[spktype],
            "id": idctr,
        }
        Input_id[input] = idctr
        Inp["nInp"] = int(nInputs_max[spktype] * nInputs_frac[spktype])

        # restrict inputs to subvolume
        Inp["nInp"], Inp["pos"] = nu.locate_GoC(
            nGoC=Inp["nInp"],
            volume=clusVolume[input],
            seed=simid * ctr + Input_id[input],
        )
        Inp["pos"] = Inp["pos"] + clusSpatialOffset[input]

        Inp["filename"] = Input_flnm[input]

        tmp = {"conn_pairs": [], "conn_wt": [], "conn_loc": []}
        currseed = simid * Inp["nInp"] * ctr + Input_id[input] + clusSpatialOffset[input][1]
        if Inp["nInp"] > 0:
            np.random.seed(currseed)
            Inp["sample"] = np.random.permutation(input_nFiles[spktype])[0 : Inp["nInp"]]  # choose wh

            (Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"],) = nu.connect_inputs_known(
                nInp=Inp["nInp"],
                Inp_pos=Inp["pos"],
                mult=Input_nRosette[spktype],
                connType=Input_conn[spktype],
                connProb=Input_prob[spktype],
                connGoC=Input_nGoC[spktype],
                connWeight=Input_wt[spktype],
                connDist=Input_maxD[spktype],
                GoC_pos=params["GoC_pos"],
                cell_loc=Input_cellloc[spktype],
                nDend=Input_dend[spktype],
                seed=currseed,
            )
            for pid in usedID:
                pairid = [x for x in range(Inp["conn_pairs"].shape[1]) if params["GoC_ParamID"][Inp["conn_pairs"][1, x]] == pid]
                tmp["conn_pairs"].append(Inp["conn_pairs"][:, pairid])
                tmp["conn_wt"].append(Inp["conn_wt"][pairid])
                tmp["conn_loc"].append(Inp["conn_loc"][:, pairid])
            for key in ["conn_pairs", "conn_wt", "conn_loc"]:
                Inp[key] = tmp[key]
            for jj in range(nGoC_pop):
                Inp["conn_pairs"][jj][1, :] = np.mod(Inp["conn_pairs"][jj][1, :], popsize)
            params["Inputs"]["types"].append(input)

        params["Inputs"][input] = Inp
        ctr += 1

    return params


if __name__ == "__main__":

    nSim = 1
    params_list = [get_simulation_params(simid) for simid in range(nSim)]

    file = open("binetwork_params.pkl", "wb")
    pkl.dump(params_list, file)
    file.close()
