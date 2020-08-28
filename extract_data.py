import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extract_helpers import *




def extract_all_temp( ):

    # add reactD once I get the files for it

    fl_arr  = [
    "./qtaim/reactC_endonf.sum",
    "./qtaim/reactC_endo10MVc.sum",
    "./qtaim/reactC_endo-10MVc.sum",
    "./qtaim/reactC_endop10MVc.sum",
    "./qtaim/reactC_endop-10MVc.sum",
    "./qtaim/reactC_endol10MVc.sum",
    "./qtaim/reactC_endol-10MVc.sum",
    "./qtaim/reactC_endoc10MVc.sum",
    "./qtaim/reactC_endoc-10MVc.sum",
    "./qtaim/reactC_exonf.sum",
    "./qtaim/reactC_exo10MVc.sum",
    "./qtaim/reactC_exo-10MVc.sum",
    "./qtaim/reactC_exop10MVc.sum",
    "./qtaim/reactC_exop-10MVc.sum",
    "./qtaim/reactC_exol10MVc.sum",
    "./qtaim/reactC_exol-10MVc.sum",
    "./qtaim/reactC_exoc10MVc.sum",
    "./qtaim/reactC_exoc-10MVc.sum"

    ]

    x = []

    for fl in fl_arr:
        if ("reactC_exo" in fl):
            atoms = [1,2,3,4,5,6]
            atom_id = ["c1","c2", "c3", "c4", "c5", "c6"]
            bond_cp = [26, 19, 21, 22, 23, 24, 25]



        if ("reactC_endo" in fl ):
            atoms = [1,3,4,5,6,7]
            atom_id = ["c1", "c3", "c4", "c5", "c6", "c7"]
            bond_cp = [26, 19, 21, 22, 23, 24, 25]

        ex1 = extract_other_crit(num=bond_cp, filename=fl)
        ex2 = extract_nuc_crit(num=atoms, filename=fl)
        ex3 = extract_charge_energies(num=atom_id, filename=fl)
        ex4 = extract_spin(num=atom_id, filename=fl)
        ex5 = extract_basics(num=atom_id, filename=fl)

        x.append([float(i) for i in ex1[1]["Stress_EigVals"][0:-1]]+ ex3 + ex4 + ex5)

    x = np.array(x)
    return x


def extract_all_csv( ):

    # add reactD once I get the files for it

    fl_arr  =[]
    x = []
    df = pd.read_excel("./barriers.xlsx")

    groups = df["group"].values.tolist()
    files = df["file"].values.tolist()
    full_name = [str(j) + "_" + str(files[i])+".sum" for i,j in enumerate(groups) ]


    df["fullname"] = full_name
    energies = np.array(df["barrier(kj/mol)"])
    print(np.mean(energies))

    #    str(df["group"]) + "_" + str(df["file"]) + ".sum"


    '''
    for fl in fl_arr:
        if ("reactC_exo" in fl):
            atoms = [1,2,3,4,5,6]
            atom_id = ["c1","c2", "c3", "c4", "c5", "c6"]
            bond_cp = [26, 19, 21, 22, 23, 24, 25]



        if ("reactC_endo" in fl ):
            atoms = [1,3,4,5,6,7]
            atom_id = ["c1", "c3", "c4", "c5", "c6", "c7"]
            bond_cp = [26, 19, 21, 22, 23, 24, 25]

        ex1 = extract_other_crit(num=bond_cp, filename=fl)
        ex2 = extract_nuc_crit(num=atoms, filename=fl)
        ex3 = extract_charge_energies(num=atom_id, filename=fl)
        ex4 = extract_spin(num=atom_id, filename=fl)
        ex5 = extract_basics(num=atom_id, filename=fl)

        x.append([float(i) for i in ex1[1]["Stress_EigVals"][0:-1]]+ ex3 + ex4 + ex5)

    x = np.array(x)
    return x

    '''
extract_all_csv()