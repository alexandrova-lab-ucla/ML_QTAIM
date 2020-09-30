import os
import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import seaborn as sns
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

    # extracts xyz position of the first diels-alder atom in the first file. Used to standardize position
    atoms = df["AtomOrder"][0]
    atoms = atoms.replace(" ","")[1:-1]
    atom_id = [x[1:-1] for x in atoms.split(",")]
    basis = extract_basics(num=atom_id, filename=fl_arr[0])
    basis_atom_1 = [basis["x_basic_1"], basis["y_basic_1"], basis["z_basic_1"]]


    for ind, fl in enumerate(fl_arr):
        #if (pd.isnull(df["AtomOrder"][ind]) or pd.isnull(df["CPOrder"][ind]) ):
        #    #print("critical points not added here")
        #    pass
        #else:
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

        translate = ["x_basic_0", "y_basic_0", "z_basic_0", "x_basic_1", "y_basic_1", "z_basic_1",
                     "x_basic_2", "y_basic_2", "z_basic_2", "x_basic_3", "y_basic_3", "z_basic_3",
                     "x_basic_4", "y_basic_4", "z_basic_4", "x_basic_5", "y_basic_5", "z_basic_5"]

        for i in translate:

            if(i[0] == 'x'):
                basics[i] = basics[i] - basis_atom_1[0]
            elif(i[0] == 'y'):
                basics[i] = basics[i] - basis_atom_1[1]
            else:
                basics[i] = basics[i] - basis_atom_1[2]

        #print(len(bond) + len(ring) + len(nuc) + len(charge)+\
        #      len(spin) + len(basics))


        full_dictionary.update(bond)
        full_dictionary.update(ring)
        full_dictionary.update(nuc)
        full_dictionary.update(charge)
        full_dictionary.update(spin)
        full_dictionary.update(basics)

        y.append(float(df["barrier(kj/mol)"][ind]))

        list_of_dicts.append(full_dictionary)
        full_dictionary = {}

    df_results = pd.DataFrame(list_of_dicts)

    return df_results, np.array(y)

x, y = extract_all()

importance_vars_v1 = [
"DelSqRho_5", "ESPn_4", "HessRho_EigVals_c_6",
"HessRho_EigVals_a_9","G_7","G_4", "GradRho_b_10",
"Kinetic_basic_4","K|Scaled|_basic_1","K|Scaled|_basic_3",
"K|Scaled|_basic_5","ESP_0","ESP_3","ESP_5","ESP_4","ESPe_9",
"NetCharge_4","NetCharge_5","NetCharge_basic_3","Rho_0",
"Rho_7","Rho_5","Stress_EigVals_c_6","Spin_tot_5",
"Vnuc_1","Vnuc_2","Vnuc_3","Vnuc_0",
"V_5","z_basic_1","z_basic_3","z_basic_4",
"z_basic_5"]

#reduced_x = x[importance_vars_v1]
#reduced_x = scale(reduced_x)
#plt.matshow(reduced_x.corr())
#plt.colorbar()
#plt.show()

#plots selected variable correlation
#reduced_x = x[importance_vars_v1]
#corr = reduced_x.corr()
#ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200), square=False)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right', fontsize='small')
#plt.show()


importance_vars_v2 = \
    ["G_7","GradRho_b_10","Kinetic_basic_4",
"K|Scaled|_basic_1","K|Scaled|_basic_3","K|Scaled|_basic_5",
"ESP_0","ESP_3","ESP_4","ESPe_9","NetCharge_4",
"NetCharge_basic_3","Rho_0","Rho_7",
"Stress_EigVals_c_6","Spin_tot_5",
"Vnuc_1","Vnuc_2","Vnuc_3","Vnuc_0",
"z_basic_1","z_basic_3","z_basic_4"]

reduced_x_2 = x[importance_vars_v2]
#corr = reduced_x_2.corr()
#ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=False)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right', fontsize='small')
#plt.show()
#print(corr)


# feature selection
x = scale(x)
#variance_thresh(x,y)
#lasso_cv(x,y)
#recursive_feat_cv(x, y)

# -------------------------------------
#pca(x)
# 15 pca components has 82% explained variance
# 20 pca components has 87% explained variance
# 25 pca components has 90% explained variance

#----------------Done and good
#lasso(x, y)
#boruta(x,y)
#recursive_feat_elim(x, y)

reduced_x_2 = scale(reduced_x_2)

pca = PCA(0.85)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principal_components)

principal_df

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


reg_bayes = BayesianRidge(n_iter=10000, tol=1e-7, copy_X=True, alpha_1=1e-03, alpha_2=1e-03,
                    lambda_1=1e-03, lambda_2=1e-03)
reg_ridge = KernelRidge(kernel='rbf', alpha=0.00005, gamma=0.0001)
reg_rf    = RandomForestRegressor()

x_train, x_test, y_train, y_test = train_test_split(principal_df, y, test_size=0.2)
sklearn_x = x_train.values
reg_bayes.fit(list(sklearn_x), y_train)
reg_ridge.fit(list(sklearn_x), y_train)
reg_rf.fit(l(sklearn_x), y_train)

score = reg_bayes.score(list(x_test), y_test)
print("bayes score:                " + str(score))
score = str(mean_squared_error(reg_bayes.predict(x_test), y_test))
print("MSE score:   " + str(score) )
score = str(mean_absolute_error(reg_bayes.predict(x_test), y_test))
print("MAE score:   " + str(score))
score = str(r2_score(reg_bayes.predict(x_test), y_test))
print("r2 score:   " + str(score))
print("................................................")
score = reg_bayes.score(list(x_test), y_test)
print("ridge score:                " + str(score))
score = str(mean_squared_error(reg_ridge.predict(x_test), y_test))
print("MSE score:   " + str(score) )
score = str(mean_absolute_error(reg_ridge.predict(x_test), y_test))
print("MAE score:   " + str(score))
score = str(r2_score(reg_ridge.predict(x_test), y_test))
print("r2 score:   " + str(score))
print("................................................")
score = reg_bayes.score(list(x_test), y_test)
print("rf score:                " + str(score))
score = str(mean_squared_error(reg_rf.predict(x_test), y_test))
print("MSE score:   " + str(score) )
score = str(mean_absolute_error(reg_rf.predict(x_test), y_test))
print("MAE score:   " + str(score))
score = str(r2_score(reg_rf.predict(x_test), y_test))
print("r2 score:   " + str(score))
