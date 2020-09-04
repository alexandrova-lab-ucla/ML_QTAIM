import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import os

def extract_all():
    # add reactD once I get the files for it
    df = pd.read_excel("../data/barriers.xlsx")

    fl_arr = []
    completed_files = []
    x = []
    y = []

    for i, j in enumerate(df["group"]):
        if (j != "reactC"):
            temp_name = "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + "-opt.sum"
            fl_arr.append(temp_name)

        else:
            temp_name = "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + ".sum"
            fl_arr.append(temp_name)


    for ind, fl in enumerate(fl_arr):

        print(fl)
        if (pd.isnull(df["AtomOrder"][ind]) or pd.isnull(df["CPOrder"][ind]) ):
            print("critical points not added here")
        else:

            completed_files.append(fl)
            y.append(df["barrier(kj/mol)"][ind])
            atoms = df["AtomOrder"][ind]
            atoms = atoms[1:-1]
            atom_id = [x.strip('\"') for x in atoms.split(",")]
            bond_cp = [int(x) for x in df["CPOrder"][ind].split(",")]

            atoms = []

            for i in atom_id:
                atoms.append(int(i[-1]))

            ex1 = extract_other_crit(num=bond_cp, filename=fl)
            ex2 = extract_nuc_crit(num=atoms, filename=fl)
            ex3 = extract_charge_energies(num=atom_id, filename=fl)
            ex4 = extract_spin(num=atom_id, filename=fl)
            ex5 = extract_basics(num=atom_id, filename=fl)

            #unpacks
            temp = []
            for i in ex1:
                for key, vals in i.items():
                    if(type(vals) == list):
                        for list_val in vals:
                            if (list_val == "NA" ):
                                temp.append(0)
                            else:
                                temp.append(float(list_val))
                    else:
                        if (vals == "NA"):
                            temp.append(0)
                        else:
                            temp.append(float(vals))

            for i in ex2:
                for key, vals in i.items():
                    if(type(vals) == list):
                        for list_val in vals:
                            if (list_val == "NA" ):
                                temp.append(0)
                            else:
                                temp.append(float(list_val))
                    else:
                        if (vals == "NA"):
                            temp.append(0)
                        else:
                            temp.append(float(vals))

            x.append(temp +  ex3 + ex4 + ex5)

    np_convert = []
    for arrays in x:
        np_convert.append(np.array(arrays))
    return np_convert, np.array(y)

x, y = extract_all()

print(np.shape(y))
print(np.shape(x))