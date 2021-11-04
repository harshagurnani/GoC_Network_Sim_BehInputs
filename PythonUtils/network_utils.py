import numpy as np
from numpy.core import multiarray
import sys

sys.path.append("../Parameters")
from scipy.spatial import distance
import math
import pickle as pkl

"""
 Helpers:
	- Distributing GoC in space
	- Setting up GJ coupling probability and conductance
"""


def locate_GoC(nGoC=0, volume=[350, 350, 80], density=4607, seed=-1):
    if seed != -1:
        np.random.seed(seed + 1000)
    x, y, z = volume
    if nGoC == 0:
        nGoC = int(density * 1e-9 * x * y * z)  # units um->mm
    GoC_pos = np.random.random([nGoC, 3]) * [x, y, z]  # in um

    return nGoC, GoC_pos


def GJ_conn(
    GoC_pos,
    prob_type="Boltzmann",
    GJw_type="Szo16_oneGJ",
    nDend=1,
    wt_k=1,
    prob_k=1.0,
    seed=-1,
):
    # Needs Pos to compute pairwise distance. nDend is morphology specific for locating GJs in detailed models.
    # modelling as single Gap junction between cells/mm3
    if seed != -1:
        np.random.seed(seed + 3000)
    N_goc = GoC_pos.shape[0]
    radDist = distance.pdist(GoC_pos, "euclidean")
    distrange = radDist
    if prob_type == "Boltzmann":
        isconn = connProb_Boltzmann(radDist / prob_k)

    elif prob_type == "Boltzmann_scaled":
        isconn = connProb_Boltzmann_scaled(radDist, prob_k)

    GJ_pairs = np.asarray(np.nonzero(isconn))
    GJ_pairs = np.asarray(
        [GJ_pairs[:, jj] for jj in range(GJ_pairs.shape[1]) if GJ_pairs[0, jj] < GJ_pairs[1, jj]]
    )  # N_pairs x 2 array = [cell1, cell2] of each pair

    radDist = distance.squareform(radDist)
    # Gap junction conductance as a function of distance
    if GJw_type == "Vervaeke2010":
        GJ_cond = set_GJ_strength_Vervaeke2010(
            np.asarray([radDist[GJ_pairs[jj, 0], GJ_pairs[jj, 1]] for jj in range(GJ_pairs.shape[0])]),
            dist_k=wt_k,
        )  # list of gj conductance for corresponding pair
        GJ_cond[GJ_cond < 0] = 0
    elif GJw_type == "Szo16_oneGJ":
        GJ_cond = set_GJ_strength_Szo2016_oneGJ(
            np.asarray([radDist[GJ_pairs[jj, 0], GJ_pairs[jj, 1]] for jj in range(GJ_pairs.shape[0])]),
            dist_k=wt_k,
        )
        GJ_cond[GJ_cond < 0] = 0

    # Numerical normalization
    if GJw_type == "Vervaeke2010":
        GJf = set_GJ_strength_Vervaeke2010(distrange, wt_k)
        GJf[GJf < 0] = 0
        GJ0 = set_GJ_strength_Vervaeke2010(distrange)
        GJ0[GJ0 < 0] = 0
    elif GJw_type == "Szo16_oneGJ":
        GJf = set_GJ_strength_Szo2016_oneGJ(distrange, wt_k)
        GJf[GJf < 0] = 0
        GJ0 = set_GJ_strength_Szo2016_oneGJ(distrange)
        GJ0[GJ0 < 0] = 0

    curr_sum = np.mean(np.sum(GJf))
    avg_sum = np.mean(np.sum(GJ0))
    GJ_cond = GJ_cond * avg_sum / curr_sum
    GJ_loc = np.c_[
        np.random.randint(nDend, size=GJ_pairs.shape),
        np.random.random(size=GJ_pairs.shape),
    ]  # col 0,1 = dend, col 2,3 = seg

    """
	# convert to nS for alex:
	GJ_connmat = np.zeros( [N_goc, N_goc] )
	GJ_connmat[GJ_pairs[:,0],GJ_pairs[:,1]]=GJ_cond
	GJ_connmat[GJ_pairs[:,1],GJ_pairs[:,0]]=GJ_cond
	GJ_connmat = np.round(GJ_connmat) * .94 #nS
	"""
    return GJ_pairs, GJ_cond, GJ_loc


def connProb_Boltzmann(radDist, seed=-1):
    # Coupling prob as Boltzmann function - from Vervaeke 2010
    if seed != -1:
        np.random.seed(seed)
    connProb = 1e-2 * (-1745 + 1836 / (1 + np.exp((radDist - 267) / 39)))
    connGen = np.random.random(radDist.shape)
    isconn = distance.squareform((connProb - connGen) > 0)  # symmetric boolean matrix with diag=0 -> GJ or not
    return isconn


def connProb_Boltzmann_scaled(radDist, probscale=1, seed=-1):
    # Coupling prob as Boltzmann function - from Vervaeke 2010
    if seed != -1:
        np.random.seed(seed)
    connProb = 1e-2 * (-1745 + 1836 / (1 + np.exp((radDist - 267) / 39))) * probscale
    connGen = np.random.random(radDist.shape)
    isconn = distance.squareform((connProb - connGen) > 0)  # symmetric boolean matrix with diag=0 -> GJ or not
    return isconn


def set_GJ_strength_Szo2016_oneGJ(radDist, dist_k=1, seed=-1):
    # Exponential fall-off of coupling coefficient, convert to GJ_cond as linear scaling
    CC = -2.3 + 29.7 * np.exp(-(radDist / dist_k) / 70.4)  # Coupling coefficient
    # GJw = np.round(2*CC/5.0)
    GJw = 2 * CC / 5.0
    return GJw


def set_GJ_strength_Vervaeke2010(radDist, dist_k=1):
    # Exponential fall-off of coupling coefficient, convert to GJ_cond
    CC = -2.3 + 29.7 * np.exp(-radDist / (70.4 * dist_k))  # Coupling coefficient

    GJw = 0.576 * np.exp(CC / 12.4) + 0.00059 * np.exp(CC / 2.79) - 0.564
    return GJw


def MF_conn(
    nMF,
    volume,
    density,
    GoC_pos,
    MF_conntype,
    MF_connprob=0.1,
    MF_connGoC=10,
    MF_wt_type="mult",
    conn_wt=1,
    seed=-1,
):
    if seed != -1:
        np.random.seed(seed + 4000)
    nMF, MF_pos = locate_GoC(nMF, volume, density)
    nGoC = GoC_pos.shape[0]
    if MF_conntype == "random_prob":
        MF_pairs = randdist_MF_syn(nMF, nGoC, pConn=MF_connprob, nConn=0)
    elif MF_conntype == "random_sample":
        MF_pairs = randdist_MF_syn(nMF, nGoC, pConn=0, nConn=MF_connGoC)
    MF_GoC_wt = get_MF_GoC_synaptic_weights(MF_pairs, MF_pos, GoC_pos, MF_wt_type, conn_wt)

    return nMF, MF_pos, MF_pairs, MF_GoC_wt


def randdist_MF_syn(nMF, nGoC, pConn=0.1, nConn=0, seed=-1):
    """
    Randomly connect a population of nMF mossy fibres to nGoC Golgi cells, wither with probability of connection pConn with iid draws (default if pConn>0). Otherwise by choosing (without replacement) nConn fibres for each GoC i.e. same number of synapses for each GoC.
    """
    if seed != -1:
        np.random.seed(seed + 5000)
    if pConn > 0:
        connGen = np.random.random((nMF, nGoC))
        isconn = connGen < pConn

    else:
        isconn = np.zeros((nMF, nGoC))
        for goc in range(nGoC):
            isconn[np.random.permutation(nMF)[0:nConn], goc] = 1

    MF_Syn_list = np.asarray(np.nonzero(isconn))
    return MF_Syn_list


def get_MF_GoC_synaptic_weights(MF_Syn_list, MF_pos, GoC_pos, method="mult", conn_wt=1):
    if method == "mult":
        syn_wt = np.ones(MF_Syn_list.shape[1], dtype=int) * conn_wt
    # elif method=='local':
    # 	syn_wt = np.ones( MF_Syn_list.shape[1] )
    return syn_wt


def find_max_conn(GoC_pos, GJ_pairs, GJ_wt, method="max_conn"):
    nGJ, net_wt, nNbr = vn.gapJuncAnalysis(GJ_pairs, GJ_wt, nGoC=GoC_pos.shape[0])
    if method == "max_conn":
        max_conn = [np.argmax(nNbr)]
    elif method == "min_conn":
        max_conn = [np.argmin(nNbr)]
    return max_conn


def get_hetero_GoC_id(nGoC, nGoCPop, densityParams, seed=-1):
    if seed != -1:
        np.random.seed(seed + 6000)

    file = open(densityParams, "rb")
    allP = pkl.load(file)
    file.close()

    series = np.random.permutation(len(allP))
    id = [allP[series[x]] for x in range(nGoCPop)]
    # id = [ allP[np.random.randint(len(allP))] for jj in range(nGoCPop) ]
    nCells_per_pop = nGoC / nGoCPop
    nGoC = nCells_per_pop * nGoCPop

    allid = []
    for jj in range(nGoCPop):
        for kk in range(nCells_per_pop):
            allid.append(id[jj])

    allid.sort()

    return allid, nGoC


def PF_conn(
    nPF,
    volume,
    PF_density,
    GoC_pos,
    PF_conntype,
    PF_connprob=0.8,
    PF_connGoC=0,
    PF_conndist=[-1],
    seed=-1,
):
    if seed != -1:
        np.random.seed(seed + 7000)
    nPF, PF_pos = locate_GoC(nPF, volume, PF_density)

    nGoC = GoC_pos.shape[0]
    if PF_conntype == "random_prob":
        PF_pairs = randdist_MF_syn(nPF, nGoC, pConn=PF_connprob, nConn=0)
    elif PF_conntype == "random_sample":
        PF_pairs = randdist_MF_syn(nPF, nGoC, pConn=0, nConn=PF_connGoC)

    if PF_conndist[0] > 0:
        # prune distal pairs
        for jj in range(3):
            if PF_conndist[jj] <= 0:
                PF_conndist[jj] = 1e9
        PF_pairs = PF_pairs[
            :,
            [
                (
                    (abs(PF_pos[PF_pairs[0, jj], 0] - GoC_pos[PF_pairs[1, jj], 0]) < PF_conndist[0])
                    & (abs(PF_pos[PF_pairs[0, jj], 1] - GoC_pos[PF_pairs[1, jj], 1]) < PF_conndist[1])
                    & (abs(PF_pos[PF_pairs[0, jj], 2] - GoC_pos[PF_pairs[1, jj], 2]) < PF_conndist[2])
                )
                for jj in range(PF_pairs.shape[1])
            ],
        ]

    return nPF, PF_pos, PF_pairs


def connect_inputs(
    maxn=0,
    frac=0,
    density=6000,
    volume=[350, 350, 80],
    mult=0,
    loc_type="random",
    connType="random_prob",
    connProb=0.5,
    connGoC=0,
    connWeight=1,
    connDist=[0],
    GoC_pos=[],
    cell_loc="soma",
    nDend=3,
    seed=-1,
):

    if frac == 0:
        return 0, [], [], [], []
    else:
        nInp, Inp_pos = locate_GoC(int(maxn * frac), volume, density, seed=seed)

    nGoC = GoC_pos.shape[0]

    ### --- to add code for rosettes

    if connType == "random_prob":
        conn_pairs = randdist_MF_syn(nInp, nGoC, pConn=connProb, nConn=connGoC)
    elif connType == "random_sample":
        conn_pairs = randdist_MF_syn(nInp, nGoC, pConn=connProb, nConn=connGoC)
    if connDist[0] > 0:
        # prune distal pairs
        if len(connDist) == 3:
            for jj in range(3):
                if connDist[jj] <= 0:
                    connDist[jj] = 1e9
            conn_pairs = conn_pairs[
                :,
                [
                    (
                        (abs(Inp_pos[conn_pairs[0, jj], 0] - GoC_pos[conn_pairs[1, jj], 0]) < connDist[0])
                        & (abs(Inp_pos[conn_pairs[0, jj], 1] - GoC_pos[conn_pairs[1, jj], 1]) < connDist[1])
                        & (abs(Inp_pos[conn_pairs[0, jj], 2] - GoC_pos[conn_pairs[1, jj], 2]) < connDist[2])
                    )
                    for jj in range(conn_pairs.shape[1])
                ],
            ]
        else:
            conn_pairs = conn_pairs[
                :,
                [
                    (
                        np.power(
                            Inp_pos[conn_pairs[0, jj], 0] - GoC_pos[conn_pairs[1, jj], 0],
                            2,
                        )
                        + np.power(
                            Inp_pos[conn_pairs[0, jj], 1] - GoC_pos[conn_pairs[1, jj], 1],
                            2,
                        )
                        + np.power(
                            Inp_pos[conn_pairs[0, jj], 2] - GoC_pos[conn_pairs[1, jj], 2],
                            2,
                        )
                    )
                    < connDist[0] ** 2
                    for jj in range(conn_pairs.shape[1])
                ],
            ]

    ### --- to add code for weight
    conn_wt = get_MF_GoC_synaptic_weights(conn_pairs, Inp_pos, GoC_pos, "mult", conn_wt=connWeight)
    conn_loc = np.r_[
        np.random.randint(nDend, size=[1, conn_pairs.shape[1]]),
        np.random.random(size=[1, conn_pairs.shape[1]]),
    ]  # col 0,1 = dend, col 2,3 = seg

    return nInp, Inp_pos, conn_pairs, conn_wt, conn_loc


def connect_inputs_known(
    nInp,
    Inp_pos,
    mult=0,
    loc_type="random",
    connType="random_prob",
    connProb=0.5,
    connGoC=0,
    connWeight=1,
    connDist=[0],
    GoC_pos=[],
    cell_loc="soma",
    nDend=3,
    seed=-1,
):

    nGoC = GoC_pos.shape[0]

    if connType == "random_prob":
        conn_pairs = randdist_MF_syn(nInp, nGoC, pConn=connProb, nConn=connGoC)
    elif connType == "random_sample":
        conn_pairs = randdist_MF_syn(nInp, nGoC, pConn=connProb, nConn=connGoC)
    if connDist[0] > 0:
        # prune distal pairs
        if len(connDist) == 3:
            for jj in range(3):
                if connDist[jj] <= 0:
                    connDist[jj] = 1e9
            conn_pairs = conn_pairs[
                :,
                [
                    (
                        (abs(Inp_pos[conn_pairs[0, jj], 0] - GoC_pos[conn_pairs[1, jj], 0]) < connDist[0])
                        & (abs(Inp_pos[conn_pairs[0, jj], 1] - GoC_pos[conn_pairs[1, jj], 1]) < connDist[1])
                        & (abs(Inp_pos[conn_pairs[0, jj], 2] - GoC_pos[conn_pairs[1, jj], 2]) < connDist[2])
                    )
                    for jj in range(conn_pairs.shape[1])
                ],
            ]
        else:
            conn_pairs = conn_pairs[
                :,
                [
                    (
                        np.power(
                            Inp_pos[conn_pairs[0, jj], 0] - GoC_pos[conn_pairs[1, jj], 0],
                            2,
                        )
                        + np.power(
                            Inp_pos[conn_pairs[0, jj], 1] - GoC_pos[conn_pairs[1, jj], 1],
                            2,
                        )
                        + np.power(
                            Inp_pos[conn_pairs[0, jj], 2] - GoC_pos[conn_pairs[1, jj], 2],
                            2,
                        )
                    )
                    < connDist[0] ** 2
                    for jj in range(conn_pairs.shape[1])
                ],
            ]

    ### --- to add code for weight
    conn_wt = get_MF_GoC_synaptic_weights(conn_pairs, Inp_pos, GoC_pos, "mult", conn_wt=connWeight)
    conn_loc = np.r_[
        np.random.randint(nDend, size=[1, conn_pairs.shape[1]]),
        np.random.random(size=[1, conn_pairs.shape[1]]),
    ]  # col 0,1 = dend, col 2,3 = seg

    return conn_pairs, conn_wt, conn_loc
