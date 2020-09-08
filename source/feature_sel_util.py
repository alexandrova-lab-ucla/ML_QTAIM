import numpy as np
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler


#######################################Method 1 #########################################33
def lasso(x, y):
    # lasso importance sampping
    scaler = StandardScaler()
    scaler.fit(x, y)
    sel = SelectFromModel(Lasso(alpha=0.1, max_iter=10000, normalize=True))
    x_train = scaler.transform(x)
    sel.fit(x, y)
    #print(sel.get_support())
    print("number of features selected via lasso: " + str(np.count_nonzero(sel.get_support())))


#######################################Method 3 #########################################33
# recursive feature elimination, tune to the number of features we want


def recursive_feat_elim(x, y):


    sgd = SGDRegressor(max_iter=100000, penalty="elasticnet", alpha=0.00001)
    rfe = RFE(estimator = sgd, n_features_to_select=40, step=1)
    rfe.fit(x,y)
    ranking = rfe.ranking_.reshape(np.shape(x)[1])
    print(ranking)

    #las = Lasso(alpha=0.1, max_iter=1000000)
    #rfe = RFE(estimator=las, n_features_to_select=20, step=1)
    #rfe.fit(x, y)
    #ranking = rfe.ranking_.reshape(np.shape(x)[1])
    #print(ranking)

    dtr = DecisionTreeRegressor()
    rfe = RFE(estimator=dtr, n_features_to_select=40, step=1)
    rfe.fit(x, y)
    ranking = rfe.ranking_.reshape(np.shape(x)[1])
    print(ranking)

    '''
    # recursive feature elimination method
    #

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
    '''

#######################################Method 2 #########################################33
# variance threshold filtering

def variance_thresh(x, y):
    print("feature length: " + str(np.shape(x)[1]))
    # ----------------------- not scaling features
    selector = VarianceThreshold()
    x_var_filter = selector.fit_transform(x)
    print("relevant features w/out min/max scaling: " + str(np.shape(x_var_filter)[1]))

    # ----------------------- min_max scale before variance filtering
    scaler = MinMaxScaler()
    scaler.fit(x)
    x_min_man = scaler.transform(x)
    selector = VarianceThreshold()
    x_var_filter = selector.fit_transform(x)
    print("relevant features with min/manx scaling: " + str(np.shape(x_var_filter)[1]))

#######################################Method 6 #########################################33
#todo: PCA

#######################################Method 4 #########################################33
#todo: Boruta

#######################################Method 5 #########################################33
#todo: 1D autoencoder
