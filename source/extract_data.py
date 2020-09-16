import os
import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.feature_selection import SelectKBest

import matplotlib.pyplot as plt


# takes: nothing
# returns: two matrices. list_of_dicts is a list of dictionaries containing
# critical values for each file. Y is the energies of each file.

def extract_all():
    # add reactD once I get the files for it
    fl_arr = []
    y = []
    list_of_dicts = []
    df = pd.read_excel("../data/barriers.xlsx")
    full_dictionary = {}

    for i, j in enumerate(df["group"]):
        if (j != "reactC"):
            temp_name = "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + "-opt.sum"
            fl_arr.append(temp_name)

        else:
            temp_name = "../sum_files/" + str(j) + "_" + str(df["file"].values[i]) + ".sum"
            fl_arr.append(temp_name)

    for ind, fl in enumerate(fl_arr):
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

            bond = extract_bond_crit(num=bond_cp, filename=fl)
            ring = extract_ring_crit(num=ring_cp, filename=fl)
            nuc = extract_nuc_crit(num=atoms, filename=fl)
            charge = extract_charge_energies(num=atom_id, filename=fl)
            spin = extract_spin(num=atom_id, filename=fl)
            basics = extract_basics(num=atom_id, filename=fl)

            full_dictionary.update(bond)
            full_dictionary.update(ring)
            full_dictionary.update(nuc)
            full_dictionary.update(charge)
            full_dictionary.update(spin)
            full_dictionary.update(basics)
            y.append(float(df["barrier(kj/mol)"][ind]))
            list_of_dicts.append(full_dictionary)
            full_dictionary = {}
    df = pd.DataFrame(list_of_dicts)
    return df, np.array(y)

x, y = extract_all()
#plt.matshow(x.corr())
#plt.colorbar()
#plt.show()
#x = scale(x)
#variance_thresh(x,y)

# -------------------------------------

#print("feature length: " + str(x[1]))
#selector = VarianceThreshold()
#x_var_filter = selector.fit_transform(x)
#lasso(x, y)
lasso_cv(x,y)
pca(x)
#boruta(x,y)
#recursive_feat_elim(x, y)
#recursive_feat_cv(x, y)
#vae(x_scaled)
#recursive_feat_elim(x_var_filter, y)