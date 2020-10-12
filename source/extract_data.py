import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import BayesianRidge, SGDRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn import gaussian_process
from sklearn.linear_model import Lasso


import seaborn as sns
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from skopt.searchcv import BayesSearchCV



# takes: nothing
# returns: two matrices. list_of_dicts is a list of dictionaries containing
# critical values for each file. Y is the energies of each file.
def score(reg, x_train, x_test, y_train, y_test, scale=1):
    print("................................................")
    try:
        score = reg.score(list(x_test), y_test)
    except:
        score = reg.score(x_test, y_test)

    print("score:                " + str(score))
    score = str(mean_squared_error(reg.predict(x_test) * scale, y_test * scale))
    print("MSE score test:   " + str(score))
    mae_test = str(mean_absolute_error(reg.predict(x_test) * scale, y_test * scale))
    print("MAE score test:   " + str(mae_test))
    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score test:   " + str(score))

    score = str(mean_squared_error(reg.predict(x_train) * scale, y_train * scale))
    print("MSE train score:   " + str(score))
    mae_train = str(mean_absolute_error(reg.predict(x_train) * scale, y_train * scale))
    print("MAE train score:   " + str(mae_train))
    score = str(r2_score(reg.predict(x_train), y_train))
    print("r2 score train:   " + str(score))

    plt.plot(y_train, reg.predict(x_train), 'o', color='black', markersize = 5)
    plt.plot(y_test, reg.predict(x_test), 'o', color='red')
    plt.text(0.5, 0.15, "MAE test: " +str(mae_test))
    plt.text(0.5, 0.25, "MAE train: " +str(mae_train))

    plt.ylabel("predicted")
    plt.ylabel("actual")
    name = str(reg.get_params()["estimator"]).split("(")[0]
    plt.title(name)
    plt.savefig(name + ".png")
    plt.clf()


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
    atoms = atoms.replace(" ", "")[1:-1]
    atom_id = [x[1:-1] for x in atoms.split(",")]
    basis = extract_basics(num=atom_id, filename=fl_arr[0])
    basis_atom_1 = [basis["x_basic_1"], basis["y_basic_1"], basis["z_basic_1"]]

    for ind, fl in enumerate(fl_arr):
        # if (pd.isnull(df["AtomOrder"][ind]) or pd.isnull(df["CPOrder"][ind]) ):
        #    #print("critical points not added here")
        #    pass
        # else:
        atoms = df["AtomOrder"][ind]
        atoms = atoms.replace(" ", "")[1:-1]
        atom_id = [x[1:-1] for x in atoms.split(",")]
        bond_cp = [int(x) for x in df["CPOrder"][ind].split(",")][6:13]
        ring_cp = bond_cp[4:6]
        bond_cp = bond_cp[0:4]

        atoms = []
        for i in atom_id:
            if (len(i) == 3):
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
                     "x_basic_4", "y_basic_4", "z_basic_5", "x_basic_5", "y_basic_5", "z_basic_5"]

        for i in translate:

            if (i[0] == 'x'):
                basics[i] = basics[i] - basis_atom_1[0]
            elif (i[0] == 'y'):
                basics[i] = basics[i] - basis_atom_1[1]
            else:
                basics[i] = basics[i] - basis_atom_1[2]

        # print(len(bond) + len(ring) + len(nuc) + len(charge)+\
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
    "HessRho_EigVals_a_9", "G_7", "G_4", "GradRho_b_10",
    "Kinetic_basic_4", "K|Scaled|_basic_1", "K|Scaled|_basic_3",
    "K|Scaled|_basic_5", "ESP_0", "ESP_3", "ESP_5", "ESP_4", "ESPe_9",
    "NetCharge_4", "NetCharge_5", "NetCharge_basic_3", "Rho_0",
    "Rho_7", "Rho_5", "Stress_EigVals_c_6", "Spin_tot_5",
    "Vnuc_1", "Vnuc_2", "Vnuc_3", "Vnuc_0",
    "V_5", "z_basic_1", "z_basic_3", "z_basic_4",
    "z_basic_5"]

# reduced_x = x[importance_vars_v1]
# reduced_x = scale(reduced_x)
# plt.matshow(reduced_x.corr())
# plt.colorbar()
# plt.show()

# plots selected variable correlation
# reduced_x = x[importance_vars_v1]
# corr = reduced_x.corr()
# ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200), square=False)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right', fontsize='small')
# plt.show()


importance_vars_v2 = \
    ["G_7", "GradRho_b_10", "Kinetic_basic_4",
     "K|Scaled|_basic_1", "K|Scaled|_basic_3", "K|Scaled|_basic_5",
     "ESP_0", "ESP_3", "ESP_4", "ESPe_9", "NetCharge_4",
     "NetCharge_basic_3", "Rho_0", "Rho_7",
     "Stress_EigVals_c_6", "Spin_tot_5",
     "Vnuc_1", "Vnuc_2", "Vnuc_3", "Vnuc_0",
     "z_basic_1", "z_basic_3", "z_basic_4"]
reduced_x_1 = x[importance_vars_v1]
reduced_x_2 = x[importance_vars_v2]
# corr = reduced_x_2.corr()
# ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=False)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right', fontsize='small')
# plt.show()
# print(corr)


# feature selection
# variance_thresh(x,y)
# lasso_cv(x,y)
# recursive_feat_cv(x, y)

# -------------------------------------
# pca(x)
# 15 pca components has 82% explained variance
# 20 pca components has 87% explained variance
# 25 pca components has 90% explained variance

# ----------------Done and good
# lasso(x, y)
# boruta(x,y)
# recursive_feat_elim(x, y)

reduced_x_2 = scale(reduced_x_2)
reduced_x_1 = scale(reduced_x_1)

pca = PCA(0.90)
principal_components = pca.fit_transform(x)

# principal_df = pd.DataFrame(data = principal_components)
# principal_df

params_bayes = {
    "n_iter": Integer(1000, 10000),
    "tol": Real(1e-9, 1e-3, prior='log-uniform'),
    "alpha_1": Real(1e-6, 1e+1, prior='log-uniform'),
    "alpha_2": Real(1e-6, 1e+1, prior='log-uniform'),
    "lambda_1": Real(1e-6, 1e+1, prior='log-uniform'),
    "lambda_2": Real(1e-6, 1e+1, prior='log-uniform')}

params_kernelridge = {"alpha": Real(1e-6, 1e0, prior='log-uniform'),
                "gamma": Real(1e-8, 1e0, prior='log-uniform')}

params_svr_lin = {"C": Real(1e-6, 1e+1, prior='log-uniform'),
                  "gamma": Real(1e-5, 1e-1, prior='log-uniform'),
                  "cache_size": Integer(500, 8000)}

params_svr_rbf = {"C": Real(1e-5, 1e+1, prior='log-uniform'),
                  "gamma": Real(1e-5, 1e-1, prior='log-uniform'),
                  "epsilon": Real(1e-2, 1e+1, prior='log-uniform'),
                  "cache_size": Integer(500, 8000)}

params_rf = {"max_depth": Integer(5, 40),
             "min_samples_split": Integer(2, 6),
             "n_estimators": Integer(300, 5000),
             "n_jobs": [mp.cpu_count()]
             }

params_nn = {"alpha": Real(1e-10, 1e-1, prior='log-uniform'),
             "max_iter": Integer(100, 10000),
             "tol": Real(1e-10, 1e1, prior='log-uniform'),
             "learning_rate_init": Real(1e-5, 1e-1, prior='log-uniform')}

params = {'l1_ratio': Real(0.1, 0.3),
          'tol': Real(1e-3, 1e-1, prior="log-uniform"),
          "epsilon": Real(1e-3, 1e0, prior="log-uniform"),
          "eta0": Real(0.01, 0.2)}

params_xgb = {
    "colsample_bytree": Real(0.5, 0.99),
    "max_depth": Integer(5, 25),
    "lambda": Real(0, 0.25),
    "learning_rate": Real(0.1, 0.25),
    "alpha": Real(0, 0.2),
    "eta": Real(0, 0.1),
    "gamma": Real(0, 0.1),
    "n_estimators": Integer(300, 5000),
    "objective": ["reg:squarederror"],
    "tree_method": ["gpu_hist"]}

params_ridge = {"tol" : Real(1e-5,1e-1,prior = "log-uniform"), "alpha": Real(1e-2, 10, prior="log-uniform")}
params_gp = {"alpha": Real(1e-12, 1e-4, prior="log-uniform")}
params_lasso = {"alpha": Real(1e-5, 1, prior="log-uniform")}

param_ada = {"n_estimators": Integer(1e1, 1e3, prior="log-uniform"),
             "learning_rate": Real(1e-2, 1e1, prior="log-uniform")}

param_extra = {"n_estimators": Integer(1e1, 1e4, prior="log-uniform"),
"min_samples_split": Integer(2,5),"min_samples_leaf" : Integer(1,2)}

param_huber = { "epsilon":Real(1.01,1.5), "alpha": Real(1e-6,1e-1, prior="log-uniform"),
                "tol": Real(1e-7,1e-2,prior="log-uniform")}

reg_lasso = Lasso()
reg_huber = HuberRegressor(max_iter = 1000)
reg_svr_rbf = SVR(kernel="rbf")
reg_svr_lin = SVR(kernel="linear")
reg_bayes = BayesianRidge()
reg_kernelridge = KernelRidge(kernel="poly", degree = 8)
reg_rf = RandomForestRegressor()
reg_sgd = SGDRegressor()
reg_nn = MLPRegressor(early_stopping = True, n_iter_no_change = 100, hidden_layer_sizes=(50,50,)
                      solver = "lbfgs")
reg_xgb = xgb.XGBRegressor()
reg_ridge = Ridge()
kernel = RBF(1.0) + 0.5 * WhiteKernel()
reg_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

reg_ada = AdaBoostRegressor()
reg_extra = ExtraTreesRegressor(criterion = "mse", bootstrap = True, ccp_alpha = 0.01)

reg_svr_rbf = BayesSearchCV(reg_svr_rbf, params_svr_rbf, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_svr_lin = BayesSearchCV(reg_svr_lin, params_svr_lin, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_bayes = BayesSearchCV(reg_bayes, params_bayes, n_iter=100, verbose=3, cv=3, n_jobs=10, scoring = "neg_mean_absolute_error")
reg_rf = BayesSearchCV(reg_rf, params_rf, n_iter=100, verbose=3, cv=3, n_jobs=10, scoring = "neg_mean_absolute_error")
reg_sgd = BayesSearchCV(reg_sgd, params, n_iter=10, verbose=3, cv=3, n_jobs=10)
reg_nn = BayesSearchCV(reg_nn, params_nn, n_iter=100, verbose=3, cv=3, n_jobs=10,  scoring = "neg_mean_absolute_error")
reg_xgb = BayesSearchCV(reg_xgb, params_xgb, n_iter=20, verbose=4, cv=3)
reg_lasso = BayesSearchCV(reg_lasso, params_lasso, n_iter=100, cv=3)
reg_ridge = BayesSearchCV(reg_ridge, params_ridge, verbose= 4, n_iter=100, cv=3)
# don't allow for selection of scoring algos
reg_gp = BayesSearchCV(reg_gp, params_gp, n_iter=100, verbose=4, cv=5)
reg_kernelridge = BayesSearchCV(reg_kernelridge, params_kernelridge, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_ada = BayesSearchCV(reg_ada, param_ada, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_extra = BayesSearchCV(reg_extra, param_extra, n_iter=250, verbose=3, cv=3, n_jobs=10)
reg_huber = BayesSearchCV(reg_huber, param_huber, n_iter=250, verbose=3, cv=3, n_jobs=10)

min = np.min(y)
max = np.max(y)
y = (y - min) / (max - min)

#ind_filtered = np.argsort(y)[10:-10]
#filt_y = y[ind_filtered]
#filt_x =  principal_components[ind_filtered]
# x_train, x_test, y_train, y_test = train_test_split(reduced_x_1, y , test_size=0.2)
#manually filter values found from other features
x_train, x_test, y_train, y_test = train_test_split(reduced_x_2, y, test_size=0.1)
#pca + filter top values
#x_train, x_test, y_train, y_test = train_test_split(filt_x, filt_y, test_size=0.1)







#reg_sgd.fit(list(x_train),y_train)
#reg_gp.fit(list(x_train),y_train)
#reg_bayes.fit(list(x_train), y_train)
#reg_ridge.fit(list(x_train), y_train)
#reg_nn.fit(list(x_train), y_train)
#reg_rf.fit(list(x_train), y_train)
#reg_ada.fit(list(x_train), y_train)
reg_extra.fit(list(x_train), y_train)
#reg_xgb.fit(x_train, y_train)
#reg_svr_lin.fit(list(x_train), y_train)
#reg_svr_rbf.fit(list(x_train), y_train)
#reg_lasso.fit(list(x_train), y_train)
#reg_sgd.fit(list(x_train),y_train)
reg_huber.fit(list(x_train), y_train)

#score(reg_sgd, x_train, x_test, y_train, y_test, max - min)
#score(reg_ridge, x_train, x_test, y_train, y_test, max - min)
#score(reg_nn, x_train, x_test, y_train, y_test, max - min)
#score(reg_bayes, x_train, x_test, y_train, y_test, max - min)
#score(reg_ada, x_train, x_test, y_train, y_test, max - min)
score(reg_extra, x_train, x_test, y_train, y_test, max - min)
score(reg_huber, x_train, x_test, y_train, y_test, max - min)
#score(reg_xgb, x_train, x_test, y_train, y_test, max - min)
#score(reg_lasso, x_train, x_test, y_train, y_test, max - min)
#score(reg_sgd, x_train, x_test, y_train, y_test, max - min)
#score(reg_svr_lin, x_train, x_test, y_train, y_test, max - min)
#score(reg_svr_rbf, x_train, x_test, y_train, y_test, max - min)
# these overfit but are good on training data
#score(reg_rf, x_train, x_test, y_train, y_test, max - min)
#score(reg_gp, x_train, x_test, y_train, y_test, max - min) # fuck with kernel
