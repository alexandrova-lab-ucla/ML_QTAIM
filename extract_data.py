import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extract_helpers import *
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



def extract_all( ):

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

barriers = pd.read_csv("./dielsalder_dataframe.csv")
y = barriers["Barrier"][6:-1].to_numpy()
x = extract_all()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)



# lasso importance sampping
scaler = StandardScaler()
scaler.fit(x_train, y_train)
print("finished")
sel = SelectFromModel(Lasso(alpha = 0.1, max_iter = 10000))
x_train = scaler.transform(x_train)
sel.fit(x_train, y_train)
print(sel.get_support())
print(np.count_nonzero(sel.get_support()))
print("finished")

# variance threshold filtering
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.5 * (1 - .5)))

# recursive feature elimination, tune to the number of features we want
svr = SVR(kernel="linear")
rfe = RFE(estimator=svr, n_features_to_select=5, step=1)
# rfe = RFECV(estimator=svr, step=1, scoring = "accuracy")
las = Lasso(alpha=0.5, max_iter=10000)
rfe = RFE(estimator=las, n_features_to_select=5, step=1)
rfe.fit(x_train, y_train)
ranking = rfe.ranking_.reshape(x[0].shape)

# print(np.shape(x))
# print(np.count_nonzero(rfe.support_))
# print(ranking)

# recursive feature elimination method
#
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import SGDRegressor, Ridge

# model choices for features - iterative
las = Lasso(alpha=0.5, max_iter=10000)
svr = SVR(kernel="linear")
dtr = DecisionTreeRegressor()
rdg = Ridge(alpha=1.0)
sgd = SGDRegressor(max_iter=100000, penalty="elasticnet", alpha=0.00001)

model_list = [las, dtr, svr, rdg, sgd]
model_list_sgd = [las, dtr, svr, rdg]
n_scores = []

for model in model_list_sgd:
    rfe = RFE(model, n_features_to_select=8, step=1)
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    cv = RepeatedKFold(n_splits=4, n_repeats=2, random_state=1)
    score_temp = cross_val_score(pipeline, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,
                                 error_score='raise')
    n_scores.append(score_temp)

names = ["lasso", "svr", "dec. tree", "Ridge", "Stochastic"]
names_sgd = ["lasso", "svr", "dec. tree", "Ridge"]

plt.boxplot(n_scores, showmeans=True, labels=names_sgd)
plt.show()

# model choice in number of features

n_scores = []
names = []
for i in range(1, 10):
    # model = Lasso(alpha=1, max_iter=10000)
    # model = DecisionTreeRegressor()
    model = SVR(kernel="linear")
    rfe = RFE(model, n_features_to_select=i, step=1)
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
    score_temp = cross_val_score(pipeline, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,
                                 error_score='raise')
    n_scores.append(score_temp)
    names.append(i)

plt.boxplot(n_scores, showmeans=True, labels=names)
plt.show()
