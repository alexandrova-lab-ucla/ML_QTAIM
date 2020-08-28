import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
from sklearn.model_selection import train_test_split


def extract_all():
    # add reactD once I get the files for it

    fl_arr = [
        "../sum_files/reactC_endonf.sum",
        "../sum_files/reactC_endo10MVc.sum",
        "../sum_files/reactC_endo-10MVc.sum",
        "../sum_files/reactC_endop10MVc.sum",
        "../sum_files/reactC_endop-10MVc.sum",
        "../sum_files/reactC_endol10MVc.sum",
        "../sum_files/reactC_endol-10MVc.sum",
        "../sum_files/reactC_endoc10MVc.sum",
        "../sum_files/reactC_endoc-10MVc.sum",
        "../sum_files/reactC_exonf.sum",
        "../sum_files/reactC_exo10MVc.sum",
        "../sum_files/reactC_exo-10MVc.sum",
        "../sum_files/reactC_exop10MVc.sum",
        "../sum_files/reactC_exop-10MVc.sum",
        "../sum_files/reactC_exol10MVc.sum",
        "../sum_files/reactC_exol-10MVc.sum",
        "../sum_files/reactC_exoc10MVc.sum",
        "../sum_files/reactC_exoc-10MVc.sum"

    ]

    x = []

    for fl in fl_arr:
        if ("reactC_exo" in fl):
            atoms = [1, 2, 3, 4, 5, 6]
            atom_id = ["c1", "c2", "c3", "c4", "c5", "c6"]
            bond_cp = [26, 19, 21, 22, 23, 24, 25]

        if ("reactC_endo" in fl):
            atoms = [1, 3, 4, 5, 6, 7]
            atom_id = ["c1", "c3", "c4", "c5", "c6", "c7"]
            bond_cp = [26, 19, 21, 22, 23, 24, 25]

        ex1 = extract_other_crit(num=bond_cp, filename=fl)
        ex2 = extract_nuc_crit(num=atoms, filename=fl)
        ex3 = extract_charge_energies(num=atom_id, filename=fl)
        ex4 = extract_spin(num=atom_id, filename=fl)
        ex5 = extract_basics(num=atom_id, filename=fl)

        x.append([float(i) for i in ex1[1]["Stress_EigVals"][0:-1]] + ex3 + ex4 + ex5)

    x = np.array(x)
    return x


barriers = pd.read_csv("../data/dielsalder_dataframe.csv")
y = barriers["Barrier"][6:-1].to_numpy()
x = extract_all()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
