import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
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

        #print(fl)
        if (pd.isnull(df["AtomOrder"][ind]) or pd.isnull(df["CPOrder"][ind]) ):
            #print("critical points not added here")
            pass
        else:

            atoms = df["AtomOrder"][ind]
            atoms = atoms.replace(" ","")[1:-1]
            atom_id = [x[1:-1] for x in atoms.split(",")]
            bond_cp = [int(x) for x in df["CPOrder"][ind].split(",")][6:13]
            ring_cp = bond_cp[4:6]
            bond_cp = bond_cp[0:4]

            atoms = []
            for i in atom_id:
                if(len(i) == 3):
                    atoms.append(int(i[1:3]))
                else:
                    atoms.append(int(i[-1]))

            bond    = extract_bond_crit(num=bond_cp, filename=fl)
            ring    = extract_ring_crit(num=ring_cp, filename=fl)
            nuc     = extract_nuc_crit(num=atoms, filename=fl)
            charge  = extract_charge_energies(num=atom_id, filename=fl)
            spin    = extract_spin(num=atom_id, filename=fl)
            basics  = extract_basics(num=atom_id, filename=fl)

            # todo: basics: rip

            if(len(bond) != 4):
                print(fl)
                print(bond_cp)
            if (len(ring) != 2):
                print(fl)
                print(ring_cp)

            #unpacks
            temp = []
            for i in bond:
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

            for i in nuc:
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

            x.append(temp +  charge + spin + basics)
            completed_files.append(fl)
            y.append(float(df["barrier(kj/mol)"][ind]))

    return x, np.array(y)

x, y = extract_all()
# -------------------------------------
print("feature length: " + str(np.shape(x)[1]))
selector = VarianceThreshold()
x_var_filter = selector.fit_transform(x)

variance_thresh(x,y)

lasso(x, y)
lasso(x_var_filter, y)

recursive_feat_elim(x, y)
#recursive_feat_elim(x_var_filter, y)