import neuroml as nml
from pyneuroml import pynml
from pyneuroml.lems import LEMSSimulation
import lems.api as lems

import os
import sys

sys.path.append("../PythonUtils/")
sys.path.append("../Parameters/")
sys.path.append("../Mechanisms/")

import numpy as np
import pickle as pkl
import math

import initialise_clustered_network_params as inp


def create_GoC_network(
    duration=1000,
    dt=0.025,
    seed=100,
    runid=0,
    mf=0.2,
    pf=0.2,
    order=[0, 1, 2, 3],
    gje=True,
    run=False,
):
    """
    mf/pf = simultaneous mf/pf inputs (fraction of mac possible)
    order = Order of modules for Loco, WMI, Pupil and WSP
    """

    ### ---------- Load Params
    p = get_params(runid=runid, mf=mf, pf=pf, order=order)

    gje_str = "wGJ" if gje else "noGJ"
    # Build network to specify cells and connectivity
    datadir = "../Data_ClusNet_" + gje_str + "/"

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    simid = (
        "behClusNet_"
        + format(runid, "05d")
        + "_mf_"
        + format(int(mf * 100), "03d")
        + "_pf_"
        + format(int(pf * 100), "03d")
        + "_order_{}{}{}{}".format(order[0], order[1], order[2], order[3])
        + "_"
        + gje_str
    )

    net = nml.Network(id=simid, type="networkWithTemperature", temperature="23 degC")
    net_doc = nml.NeuroMLDocument(id=net.id)
    net_doc.networks.append(net)

    arr_flnm = [
        "../Mechanisms/SpikeArray_MFON_dur_060_080.nml",
        "../Mechanisms/SpikeArray_Loco_dur_040.nml",
        "../Mechanisms/SpikeArray_Loco_PF_dur_040.nml",
        "../Mechanisms/SpikeArray_WMIAmp_dur_040.nml",
        "../Mechanisms/SpikeArray_WMIAmp_PF_dur_040.nml",
        "../Mechanisms/SpikeArray_Pupil_dur_040.nml",
        "../Mechanisms/SpikeArray_Pupil_PF_dur_040.nml",
        "../Mechanisms/SpikeArray_WSP_dur_040.nml",
        "../Mechanisms/SpikeArray_WSP_PF_dur_040.nml",
    ]

    ### -------------- Component types ------------------------- ###

    # ---- 1. GoC types
    goc_params = np.unique(np.asarray(p["GoC_ParamID"]))

    ctr = 0
    goc_type = []
    for pid in goc_params:
        gocID = "GoC_" + format(pid, "05d")
        goc_type.append(gocID)
        goc_filename = "../Cells/Golgi/{}.cell.nml".format(goc_type[ctr])
        goc_type[ctr] = pynml.read_neuroml2_file(goc_filename).cells[0]
        net_doc.includes.append(nml.IncludeType(href=goc_filename))
        ctr += 1

    # ---- 2. Inputs
    synapse = {}
    inputGen = {}
    inputBG = {}
    inputBeh = {}

    for input in p["Inputs"]["types"]:
        # Load synapse type
        Inp = p["Inputs"][input]
        net_doc.includes.append(nml.IncludeType(href=Inp["syn_type"][0]))  # filename
        if Inp["syn_type"][1] == "ExpThreeSynapse":
            synapse[input] = pynml.read_neuroml2_file(Inp["syn_type"][0]).exp_three_synapses[0]  # Component Type
        elif Inp["syn_type"][1] == "ExpTwoSynapse":
            synapse[input] = pynml.read_neuroml2_file(Inp["syn_type"][0]).exp_two_synapses[0]

        # Set up spike generators
        if Inp["type"] == "switchPoisson":  # 2 Rates: from 0 to delay, and delay to delay+duration
            inputGen[input] = pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml").components["Del_" + format(Inp["rate"][0])].id

        elif Inp["type"] == "MF_step":  # 2 Rates: from 0 to delay, and delay to delay+duration
            inputGen[input] = pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml").components["MF" + format(Inp["id"])].id

        elif Inp["type"] == "PF_step":  # 2 Rates: from 0 to delay, and delay to delay+duration
            inputGen[input] = pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml").components["PF" + format(Inp["id"])].id

        elif Inp["type"] == "MFON":  # beh - ON
            print("MFON")
            inputBeh[input] = "MFON"

        elif Inp["type"] == "PFON":  # beh - ON
            print("PFON")
            inputBeh[input] = "PFON"

        elif Inp["type"] == "MFOFF":  # beh - OFF
            print("MFOFF")
            inputBeh[input] = "MFOFF"

        elif Inp["type"] == "PFOFF":  # beh - OFF
            print("PFOFF")
            inputBeh[input] = "PFOFF"

        elif Inp["type"] == "MFRand":  # beh - OFF
            print("MFRand")
            inputBeh[input] = "MFRand"

        elif Inp["type"] == "PFRand":  # beh - OFF
            print("PFRand")
            inputBeh[input] = "PFRand"

        elif Inp["type"] == "BehMF":  # Single beh dominated MF
            print(input, Inp["type"])
            inputBeh[input] = input

        elif Inp["type"] == "BehPF":  # Single beh dominated PF
            print(input, Inp["type"])
            inputBeh[input] = input

        elif Inp["type"] == "transientPoisson":
            inputGen[input] = pynml.read_lems_file("../Mechanisms/lems_instances_ON.xml").components["Transient_350_2s_100ms"].id

        elif Inp["type"] == "poisson":
            inputBG[input] = nml.SpikeGeneratorPoisson(id=input, average_rate="{} Hz".format(Inp["rate"][0]))
            net_doc.spike_generator_poissons.append(inputBG[input])

        elif Inp["type"] == "constantRate":
            inputBG[input] = nml.SpikeGenerator(id=input, period="{} ms".format(1000.0 / (Inp["rate"][0])))
            net_doc.spike_generators.append(inputBG[input])

    # ---- 3.  Gap Junctions
    gj = nml.GapJunction(id="GJ_0", conductance="426pS")  # GoC synapse
    net_doc.gap_junctions.append(gj)

    ### ------------------ Populations ------------------------- ###

    # Create GoC population
    goc_pop = []
    goc = 0
    for pid in range(p["nPop"]):
        goc_pop.append(
            nml.Population(
                id=goc_type[pid].id + "Pop",
                component=goc_type[pid].id,
                type="populationList",
                size=p["nGoC_per_pop"],
            )
        )
        for ctr in range(p["nGoC_per_pop"]):
            inst = nml.Instance(id=ctr)
            goc_pop[pid].instances.append(inst)
            inst.location = nml.Location(x=p["GoC_pos"][goc, 0], y=p["GoC_pos"][goc, 1], z=p["GoC_pos"][goc, 2])
            goc += 1
        net.populations.append(goc_pop[pid])

    ### Background Input population
    inputBG_pop = {}
    for input in inputBG:
        Inp = p["Inputs"][input]
        inputBG_pop[input] = nml.Population(
            id=inputBG[input].id + "_pop",
            component=inputBG[input].id,
            type="populationList",
            size=Inp["nInp"],
        )
        for ctr in range(Inp["nInp"]):
            inst = nml.Instance(id=ctr)
            inputBG_pop[input].instances.append(inst)
            inst.location = nml.Location(x=Inp["pos"][ctr, 0], y=Inp["pos"][ctr, 1], z=Inp["pos"][ctr, 2])
        net.populations.append(inputBG_pop[input])

    # MF/PF as spike array
    ctr = 1000
    InpProj = {}

    inputGen_pop = {}
    allID = {}
    # Add spike array populations
    for input in inputBeh:
        Inp = p["Inputs"][input]
        inputGen_pop[input] = []
        allID[input] = {}

        for mfii in range(Inp["nInp"]):
            arrid = Inp["sample"][mfii]
            currinp = pynml.read_neuroml2_file(Inp["filename"]).spike_arrays[arrid]
            allID[input][mfii] = currinp.id
            inputGen_pop[input].append(
                nml.Population(
                    id=input + "_" + currinp.id + "_pop",
                    component=currinp.id,
                    type="populationList",
                    size=1,
                )
            )
            inst = nml.Instance(id=0)
            inst.location = nml.Location(x=Inp["pos"][mfii, 0], y=Inp["pos"][mfii, 1], z=Inp["pos"][mfii, 2])
            inputGen_pop[input][mfii].instances.append(inst)
            net.populations.append(inputGen_pop[input][mfii])

    ### ------------ Connectivity

    ### 1. Background  inputs:
    dend_id = [1, 2, 5]
    nDend = 3

    BG_Proj = {}
    for input in inputBG:
        Inp = p["Inputs"][input]
        BG_Proj[input] = []

        for jj in range(p["nPop"]):
            BG_Proj[input].append(
                nml.Projection(
                    id="{}_to_{}".format(input, goc_type[jj].id),
                    presynaptic_population=inputBG_pop[input].id,
                    postsynaptic_population=goc_pop[jj].id,
                    synapse=synapse[input].id,
                )
            )
            net.projections.append(BG_Proj[input][jj])
            ctr = 0
            nSyn = Inp["conn_pairs"][jj].shape[1]
            for syn in range(nSyn):
                pre, goc = Inp["conn_pairs"][jj][:, syn]
                if Inp["syn_loc"] == "soma":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(inputBG_pop[input].id, pre, inputBG[input].id),
                        post_cell_id="../{}/{}/{}".format(goc_pop[jj].id, goc, goc_type[jj].id),
                        post_segment_id="0",
                        post_fraction_along="0.5",
                        weight=Inp["conn_wt"][jj][syn],
                        delay="0 ms",
                    )  # on soma
                elif Inp["syn_loc"] == "dend":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(inputBG_pop[input].id, pre, inputBG[input].id),
                        post_cell_id="../{}/{}/{}".format(goc_pop[jj].id, goc, goc_type[jj].id),
                        post_segment_id=dend_id[int(Inp["conn_loc"][jj][0, syn])],
                        post_fraction_along=dend_id[int(Inp["conn_loc"][jj][1, syn])],
                        weight=Inp["conn_wt"][jj][syn],
                        delay="0 ms",
                    )  # on dend
                BG_Proj[input][jj].connection_wds.append(conn)
                ctr += 1

    InputGen_Proj = {}
    # spike arrays
    for input in inputBeh:
        Inp = p["Inputs"][input]
        InputGen_Proj[input] = []

        ctr = 0
        for jj in range(p["nPop"]):
            InputGen_Proj[input].append([])
            # initialise all projections
            for mfii in range(Inp["nInp"]):
                InputGen_Proj[input][jj].append(
                    nml.Projection(
                        id="{}_{}_to_{}".format(input, mfii, goc_type[jj].id),
                        presynaptic_population=inputGen_pop[input][mfii].id,
                        postsynaptic_population=goc_pop[jj].id,
                        synapse=synapse[input].id,
                    )
                )
                net.projections.append(InputGen_Proj[input][jj][mfii])

            nSyn = Inp["conn_pairs"][jj].shape[1]
            for syn in range(nSyn):
                pre, goc = Inp["conn_pairs"][jj][:, syn]
                if Inp["syn_loc"] == "soma":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(inputGen_pop[input][pre].id, 0, allID[input][pre]),
                        post_cell_id="../{}/{}/{}".format(goc_pop[jj].id, goc, goc_type[jj].id),
                        post_segment_id="0",
                        post_fraction_along="0.5",
                        weight=Inp["conn_wt"][jj][syn],
                        delay="{} ms".format(np.random.random()),
                    )  # on soma
                elif Inp["syn_loc"] == "dend":
                    conn = nml.ConnectionWD(
                        id=ctr,
                        pre_cell_id="../{}/{}/{}".format(inputGen_pop[input][pre].id, 0, allID[input][pre]),
                        post_cell_id="../{}/{}/{}".format(goc_pop[jj].id, goc, goc_type[jj].id),
                        post_segment_id=dend_id[int(Inp["conn_loc"][jj][0, syn])],
                        post_fraction_along=dend_id[int(Inp["conn_loc"][jj][1, syn])],
                        weight=Inp["conn_wt"][jj][syn],
                        delay="{} ms".format(np.random.random()),
                    )  # on dend
                InputGen_Proj[input][jj][pre].connection_wds.append(conn)
                ctr += 1

    ### 3. Electrical coupling between GoCs

    GoCCoupling = []
    ctr = 0
    if gje:
        print("Adding electrical connections...")
        for pre in range(p["nPop"]):
            for post in range(pre, p["nPop"]):
                GoCCoupling.append(
                    nml.ElectricalProjection(
                        id="GJ_{}_{}".format(goc_pop[pre].id, goc_pop[post].id),
                        presynaptic_population=goc_pop[pre].id,
                        postsynaptic_population=goc_pop[post].id,
                    )
                )
                net.electrical_projections.append(GoCCoupling[ctr])

                gjParams = p["econn_pop"][pre][post - pre]
                for jj in range(gjParams["GJ_pairs"].shape[0]):
                    conn = nml.ElectricalConnectionInstanceW(
                        id=jj,
                        pre_cell="../{}/{}/{}".format(
                            goc_pop[pre].id,
                            gjParams["GJ_pairs"][jj, 0],
                            goc_type[pre].id,
                        ),
                        pre_segment=dend_id[int(gjParams["GJ_loc"][jj, 0])],
                        pre_fraction_along="0.5",
                        post_cell="../{}/{}/{}".format(
                            goc_pop[post].id,
                            gjParams["GJ_pairs"][jj, 1],
                            goc_type[post].id,
                        ),
                        post_segment=dend_id[int(gjParams["GJ_loc"][jj, 1])],
                        post_fraction_along="0.5",
                        synapse=gj.id,
                        weight=gjParams["GJ_wt"][jj],
                    )
                    GoCCoupling[ctr].electrical_connection_instance_ws.append(conn)
                ctr += 1

    print("Writing files...")
    ### --------------  Write files

    for file in arr_flnm:
        net_doc.includes.append(nml.IncludeType(href=file))

    net_filename = simid + ".net.nml"  # NML2 description of network instance
    pynml.write_neuroml2_file(net_doc, net_filename)

    simid = "sim_" + net.id
    ls = LEMSSimulation(simid, duration=duration, dt=dt, simulation_seed=seed)
    ls.assign_simulation_target(net.id)
    ls.include_neuroml2_file(net_filename)  # Network NML2 file
    ls.include_lems_file("../Mechanisms/inputGenerators.xml")
    ls.include_lems_file("../Mechanisms/lems_instances_ON.xml")

    # Specify outputs
    eof0 = "Events_file"
    ls.create_event_output_file(eof0, datadir + "%s.spikes.dat" % simid, format="ID_TIME")
    ctr = 0
    for pid in range(p["nPop"]):
        for jj in range(goc_pop[pid].size):
            ls.add_selection_to_event_output_file(
                eof0,
                ctr,
                "{}/{}/{}".format(goc_pop[pid].id, jj, goc_type[pid].id),
                "spike",
            )
            ctr += 1

    eof1 = "Events_file2"
    ls.create_event_output_file(eof1, datadir + "%s.inputs.dat" % simid, format="ID_TIME")
    ctr = 0
    for input in inputBeh:
        for pid in range(len(inputGen_pop[input])):
            ls.add_selection_to_event_output_file(
                eof1,
                ctr,
                "{}/{}/{}".format(inputGen_pop[input][pid].id, 0, allID[input][pid]),
                "spike",
            )
            ctr += 1

    """
	of0 = 'Volts_file'
	ls.create_output_file(of0, datadir+"%s.v.dat"%simid)
	ctr=0
	for pid in range( p["nPop"] ):
		for jj in range( goc_pop[pid].size ):
			ls.add_column_to_output_file(of0, ctr, '{}/{}/{}/v'.format( goc_pop[pid].id, jj, goc_type[pid].id))
			ctr +=1

	"""

    # Create Lems file to run
    lems_simfile = ls.save_to_file()

    if run:
        res = pynml.run_lems_with_jneuroml_neuron(lems_simfile, max_memory="2G", nogui=True, plot=False)
    else:
        res = pynml.run_lems_with_jneuroml_neuron(
            lems_simfile,
            max_memory="2G",
            only_generate_scripts=True,
            compile_mods=False,
            nogui=True,
            plot=False,
        )

    res = True
    return res


def get_params(runid=0, mf=0.2, pf=0.2, order=[0, 1, 2, 3]):

    Input_prob = {"MF_bg": 0.3, "PF_bg": 0.5, "BehMF": 0.2, "BehPF": 0.2}

    offset = order * 125
    clusSpatialOffset = {
        "LocoMF": [0, offset[0], 0],
        "LocoPF": [0, offset[0], 0],
        "WMIAmpMF": [0, offset[1], 0],
        "WMIAmpPF": [0, offset[1], 0],
        "PupilMF": [0, offset[2], 0],
        "PupilPF": [0, offset[2], 0],
        "WSPMF": [0, offset[3], 0],
        "WSPPF": [0, offset[3], 0],
    }

    nMF = 60
    nPF = 120

    nPop = 5

    nInputs_max = {"MF_bg": nMF, "PF_bg": nPF, "BehMF": nMF, "BehPF": nPF}
    nInputs_frac = {"MF_bg": 1.0, "PF_bg": 1.0, "BehMF": mf, "BehPF": pf}

    params = inp.get_simulation_params(
        runid,
        nInputs_max=nInputs_max,
        nInputs_frac=nInputs_frac,
        nGoC_pop=nPop,
        Input_prob=Input_prob,
        stepInputs=[],
        behInputs=[],
        clusSpatialOffset=clusSpatialOffset,
    )
    return params


if __name__ == "__main__":
    runid = 0

    print("Generating network using parameters for runid=", runid)
    res = create_GoC_network(duration=20000, dt=0.025, seed=123, runid=runid)
    print(res)
