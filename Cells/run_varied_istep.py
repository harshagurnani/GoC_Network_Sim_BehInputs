import neuroml as nml
from pyneuroml import pynml
from pyneuroml.lems import LEMSSimulation
import lems.api as lems

import numpy as np
import pickle as pkl
import sys


def create_GoC_network(
    duration=2000,
    dt=0.025,
    seed=123,
    runid=0,
    run=False,
    minI=-75,
    maxI=200,
    iStep=25,
    iDur=400,
    iRest=500,
    girk=False,
):

    wGirk = ""
    if girk:
        wGirk = "_wGIRK"
    ### ---------- Component types
    gocID = "GoC_" + format(runid, "05d") + wGirk
    goc_filename = "{}.cell.nml".format(gocID)
    goc_type = pynml.read_neuroml2_file(goc_filename).cells[0]

    ### --------- Populations

    # Build network to specify cells and connectivity
    net = nml.Network(
        id="GoCNet_" + format(runid, "05d"),
        type="networkWithTemperature",
        temperature="23 degC",
    )

    # Create GoC population
    goc_pop = nml.Population(id=goc_type.id + "Pop", component=goc_type.id, type="populationList", size=1)
    inst = nml.Instance(id=0)
    goc_pop.instances.append(inst)
    inst.location = nml.Location(x=0, y=0, z=0)
    net.populations.append(goc_pop)

    # Create NML document for network specification
    net_doc = nml.NeuroMLDocument(id=net.id)
    net_doc.networks.append(net)
    net_doc.includes.append(nml.IncludeType(href=goc_filename))

    # Add Current Injection
    ctr = 0
    goc = 0
    p = {
        "iAmp": np.arange(minI, maxI + iStep / 2, iStep),
        "iDuration": iDur,
        "iRest": iRest,
    }
    p["nSteps"] = p["iAmp"].shape[0]

    for jj in range(p["nSteps"]):
        input_id = "stim_{}".format(ctr)
        istep = nml.PulseGenerator(
            id=input_id,
            delay="{} ms".format(p["iDuration"] * jj + p["iRest"] * (jj + 1)),
            duration="{} ms".format(p["iDuration"]),
            amplitude="{} pA".format(p["iAmp"][jj]),
        )
        net_doc.pulse_generators.append(istep)

        input_list = nml.InputList(id="ilist_{}".format(ctr), component=istep.id, populations=goc_pop.id)
        curr_inj = nml.Input("0", target="../%s[%i]" % (goc_pop.id, goc), destination="synapses")
        input_list.input.append(curr_inj)
        net.input_lists.append(input_list)
        ctr += 1

    ### --------------  Write files

    net_filename = "GoCNet_istep_" + format(runid, "05d") + ".nml"
    pynml.write_neuroml2_file(net_doc, net_filename)

    simid = "sim_gocnet_istep_" + goc_type.id
    ls = LEMSSimulation(simid, duration=duration, dt=dt, simulation_seed=seed)
    ls.assign_simulation_target(net.id)
    ls.include_neuroml2_file(net_filename)
    ls.include_neuroml2_file(goc_filename)

    # Specify outputs
    eof0 = "Events_file"
    ls.create_event_output_file(eof0, "%s.v.spikes" % simid, format="ID_TIME")
    for jj in range(goc_pop.size):
        ls.add_selection_to_event_output_file(eof0, jj, "{}/{}/{}".format(goc_pop.id, jj, goc_type.id), "spike")

    of0 = "Volts_file"
    ls.create_output_file(of0, "%s.v.dat" % simid)
    for jj in range(goc_pop.size):
        ls.add_column_to_output_file(of0, jj, "{}/{}/{}/v".format(goc_pop.id, jj, goc_type.id))

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

    return res


if __name__ == "__main__":
    runid = 0
    if len(sys.argv) > 1:
        runid = int(sys.argv[1])
    print("Generating network using parameters for runid=", runid)
    res = create_GoC_network(duration=13000, dt=0.025, seed=123, runid=runid)
    print(res)
