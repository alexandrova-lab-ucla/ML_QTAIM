import os
import seaborn as sns
import multiprocessing as mp
import numpy as np
import pandas as pd
from extract_helpers import *
from feature_sel_util import *
import xgboost as xgb

import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from skopt.searchcv import BayesSearchCV

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.linear_model import BayesianRidge, SGDRegressor, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor


class custom_skopt_extra_scorer(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, res):

        param_extra = {"n_estimators": res["x"][2],
                       "min_samples_split": res["x"][1],
                        "min_samples_leaf": res["x"][0]}

        reg = ExtraTreesRegressor(**param_extra, criterion = "mae")
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        reg.fit(x_train, y_train)
        y_pred = np.array(reg.predict(x_test))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if (mae < 10):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(param_extra)
        return 0

class custom_skopt_xgb_scorer(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, res):

        dict = {
            "objective":"reg:squarederror", "tree_method":"gpu_hist",
             "alpha":res["x"][0], "colsample_bytree":res["x"][1],"eta":res["x"][2],
            "gamma":res["x"][3], "lambda": res["x"][4], "learning_rate":res["x"][5],
            "max_depth":res["x"][6],"n_estimators":res["x"][7]
        }
        reg = xgb.XGBRegressor(**dict)

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        reg.fit(x_train, y_train)
        y_pred = np.array(reg.predict(x_test))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if (mae < 10):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(param_extra)
        return 0

class custom_skopt_rf_scorer(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, res):
        params_rf = {"max_depth": res["x"][2],
                     "min_samples_split": res["x"][1],
                     "n_estimators": res["x"][0]
                     }

        reg = RandomForestRegressor(**params_rf, n_jobs= -1)

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        reg.fit(x_train, y_train)
        y_pred = np.array(reg.predict(x_test))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if (mae < 10):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(param_extra)
        return 0

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


x, y = extract_all()

min = np.min(y)
max = np.max(y)
y_scale = (y - min) / (max - min)

importance_vars_v1 = [
    "DelSqV_6",
    "ESP_0", "ESP_4", "ESP_5",
    "ESPe_4", "ESPe_9", "ESPn_4",
    "G_0", "G_4", "GradRho_b_10",
    "HessRho_EigVals_a_9", "HessRho_EigVals_b_9", "HessRho_EigVals_c_6",
    "K|Scaled|_basic_1", "K|Scaled|_basic_3",
    "Lagr_basic_0",
    "NetCharge_5", "NetCharge_basic_1",
    "Rho_0",
    "Stress_EigVals_c_6",
    "Spin_tot_5",
    "V_9",  "Vnuc_0", "Vnuc_1", "Vnuc_2", "Vnuc_3", "Vnuc_9",
    "x_basic_4", "x_basic_5", "z_basic_3", "z_basic_4", "z_basic_5"]
importance_vars_v3 = \
    [
    "ESP_0","ESP_3","ESP_1","ESP_2","ESP_4","ESP_5","ESPe_9","ESPn_4",
    "G_0","G_4","G_9",
    "HessRho_EigVals_a_9","HessRho_EigVals_b_9","HessRho_EigVals_c_6",
    "K|Scaled|_basic_3","Kinetic_basic_4","Lagr_basic_0","Lagr_basic_3",
    "NetCharge_5",
    "NetCharge_basic_1","NetCharge_basic_3","NetCharge_basic_4",
    "Rho_0","Spin_tot_4","Spin_tot_5",
    "Stress_EigVals_c_6","V_9","Vnuc_0","Vnuc_1","Vnuc_2",
    "x_basic_4","y_basic_3","z_basic_4","z_basic_5"]

importance_vars_v2 = \
    [
        "DelSqV_6",
        "ESP_0", "ESP_4", "ESP_5",
        "ESPe_4",
        "G_0", "GradRho_b_10",
        "K|Scaled|_basic_1", "K|Scaled|_basic_3",
        "Lagr_basic_0",
        "NetCharge_5", "NetCharge_basic_1",
        "Rho_0",
        "Vnuc_0", "Vnuc_1", "Vnuc_2", "Vnuc_3",
        "x_basic_4", "x_basic_5", "z_basic_3", "z_basic_4", "z_basic_5"]

importance_vars_v4 = \
    [
    "ESP_0","ESP_3","ESP_1","ESP_2","ESP_4","ESP_5","ESPn_4",
    "G_0","G_9",
    "K|Scaled|_basic_3","Kinetic_basic_4","Lagr_basic_0","Lagr_basic_3",
    "NetCharge_basic_1","NetCharge_basic_3","NetCharge_basic_4",
    "Rho_0","Spin_tot_4","Spin_tot_5",
    "Vnuc_0","Vnuc_1","Vnuc_2",
    "z_basic_4","z_basic_5"]

# plots selected variable correlation
reduced_x = x[importance_vars_v3]
corr = reduced_x.corr()
ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200), square=False)
ax.set_xticklabels(ax.get_xticklabels(),rotation=70, horizontalalignment='right', fontsize='x-small')
plt.show()

reduced_x = x[importance_vars_v4]
corr = reduced_x.corr()
ax = sns.heatmap(corr,  vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200), square=False)
ax.set_xticklabels(ax.get_xticklabels(),rotation=70, horizontalalignment='right', fontsize='x-small')
plt.show()


reduced_x_1 = x[importance_vars_v1]
reduced_x_2 = x[importance_vars_v2]
reduced_x_3 = x[importance_vars_v3]
reduced_x_4 = x[importance_vars_v4]
reduced_x_4 = scale(reduced_x_4)
reduced_x_3 = scale(reduced_x_3)
reduced_x_2 = scale(reduced_x_2)
reduced_x_1 = scale(reduced_x_1)

#-------------------------feature selection
# variance_thresh(x,y)
# -------------------------------------
#pca(x)
# 15 pca components has 82% explained variance
# 20 pca components has 87% explained variance
# 25 pca components has 90% explained variance
# ----------------Done and good
#lasso(x, y)
#boruta(x,y, n = 7)
#boruta(x,y, n = 5)
#boruta(x,y, n = 3)

#recursive_feat_elim(x, y)
#pca = PCA(0.90)
#principal_components = pca.fit_transform(x)
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
param_knn = {"n_neighbors": Integer(3,7)
}

reg_lasso = Lasso()
reg_svr_rbf = SVR(kernel="rbf")
reg_svr_lin = SVR(kernel="linear")
reg_bayes = BayesianRidge()
reg_kernelridge = KernelRidge(kernel="poly", degree = 8)
reg_rf = RandomForestRegressor()
reg_sgd = SGDRegressor()
reg_ridge = Ridge()
reg_xgb = xgb.XGBRegressor()
kernel = RBF(1.0) + 0.5 * WhiteKernel()
reg_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

reg_nn = MLPRegressor(early_stopping = True, n_iter_no_change = 100, hidden_layer_sizes=(50,50,),
                      solver = "lbfgs")
reg_ada = AdaBoostRegressor()
reg_extra = ExtraTreesRegressor(criterion = "mae")
reg_huber = HuberRegressor(max_iter = 1000)
reg_knn = KNeighborsRegressor(algorithm = "auto", weights = "distance")



#ind_filtered = np.argsort(y)[10:-10]
#filt_y = y[ind_filtered]
#filt_x =  principal_components[ind_filtered]
# x_train, x_test, y_train, y_test = train_test_split(reduced_x_1, y , test_size=0.2)
#manually filter values found from other features
x_train, x_test, y_train, y_test = train_test_split(reduced_x_4, y_scale , test_size=0.2)

''' # tensorflow
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(reduced_x_2.shape[0], )),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])
model.fit(x_train,  np.ravel(y_train), epochs=100, verbose = 1)
# tensorflow 

y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
y_pred = model.predict(x_train)
print(confusion_matrix(y_train, y_pred))
'''
#-------------------------------------------------



#pca + filter top values
#x_train, x_test, y_train, y_test = train_test_split(filt_x, filt_y, test_size=0.1)


reg_svr_rbf = BayesSearchCV(reg_svr_rbf, params_svr_rbf, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_svr_lin = BayesSearchCV(reg_svr_lin, params_svr_lin, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_bayes = BayesSearchCV(reg_bayes, params_bayes, n_iter=100, verbose=3, cv=3, n_jobs=10, scoring = "neg_mean_absolute_error")
reg_rf = BayesSearchCV(reg_rf, params_rf, n_iter=250, verbose=3, cv=3, n_jobs=10, scoring = "neg_mean_absolute_error")
reg_sgd = BayesSearchCV(reg_sgd, params, n_iter=10, verbose=3, cv=3, n_jobs=10)
reg_xgb = BayesSearchCV(reg_xgb, params_xgb, n_iter=20, verbose=4, cv=3)
reg_lasso = BayesSearchCV(reg_lasso, params_lasso, n_iter=100, cv=3)
reg_ridge = BayesSearchCV(reg_ridge, params_ridge, verbose= 4, n_iter=100, cv=3)
# don't allow for selection of scoring algos
reg_gp = BayesSearchCV(reg_gp, params_gp, n_iter=25, verbose=4, cv=5)
reg_kernelridge = BayesSearchCV(reg_kernelridge, params_kernelridge, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_ada = BayesSearchCV(reg_ada, param_ada, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_nn = BayesSearchCV(reg_nn, params_nn, n_iter=10, verbose=3, cv=3, n_jobs=10,  scoring = "neg_mean_absolute_error")
reg_extra = BayesSearchCV(reg_extra, param_extra, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_huber = BayesSearchCV(reg_huber, param_huber, n_iter=100, verbose=3, cv=3, n_jobs=10)
reg_knn = BayesSearchCV(reg_knn, param_knn, n_iter=10, verbose=3, cv=3, n_jobs=10)

custom_scorer_extra = custom_skopt_extra_scorer
custom_scorer_rf = custom_skopt_rf_scorer
custom_scorer_xgb = custom_skopt_xgb_scorer

#reg_sgd.fit(list(x_train),y_train)
#reg_gp.fit(list(x_train),y_train)
#reg_bayes.fit(list(x_train), y_train)
#reg_ridge.fit(list(x_train), y_train)
#reg_ada.fit(list(x_train), y_train)
#reg_xgb.fit(x_train, y_train,callback=[custom_skopt_xgb_scorer(x,y)])
#reg_svr_lin.fit(list(x_train), y_train)
#reg_svr_rbf.fit(list(x_train), y_train)
#reg_lasso.fit(list(x_train), y_train)
#reg_sgd.fit(list(x_train),y_train)
#reg_nn.fit(list(x_train), y_train)
#reg_rf.fit(list(x_train), y_train, callback=[custom_skopt_rf_scorer(x,y)])
reg_extra.fit(list(x_train), y_train, callback=[custom_scorer_extra(x,y)])
#reg_huber.fit(list(x_train), y_train)
#reg_knn.fit(list(x_train), y_train)

#score(reg_sgd, x_train, x_test, y_train, y_test, max - min)
#score(reg_ridge, x_train, x_test, y_train, y_test, max - min)
#score(reg_bayes, x_train, x_test, y_train, y_test, max - min)
#score(reg_ada, x_train, x_test, y_train, y_test, max - min)
#score(reg_nn, x_train, x_test, y_train, y_test, max - min)
#score(reg_huber, x_train, x_test, y_train, y_test, max - min)
#score(reg_knn, x_train, x_test, y_train, y_test, max - min)
#score(reg_xgb, x_train, x_test, y_train, y_test, max - min)
#score(reg_lasso, x_train, x_test, y_train, y_test, max - min)
#score(reg_sgd, x_train, x_test, y_train, y_test, max - min)
#score(reg_svr_lin, x_train, x_test, y_train, y_test, max - min)
#score(reg_svr_rbf, x_train, x_test, y_train, y_test, max - min)
#score(reg_rf, x_train, x_test, y_train, y_test, max - min)
score(reg_extra, x_train, x_test, y_train, y_test, max - min)
#score(reg_gp, x_train, x_test, y_train, y_test, max - min) # fuck with kernel



