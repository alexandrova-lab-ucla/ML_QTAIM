import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from boruta import BorutaPy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, RFECV, \
    SelectFromModel, VarianceThreshold
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor

#######################################Method 1 #########################################33
def lasso(x, y):
    # lasso importance sampping
    x_scaled = scale(x)
    sel = SelectFromModel(Lasso(alpha=0.9, max_iter=20000, normalize = True))
    sel.fit(x_scaled, y)
    print("number of features selected via lasso: " + str(np.count_nonzero(sel.get_support())))
    for i, j in enumerate(sel.get_support()):
        if j != 0:
            print(x.columns.values[i])

def lasso_cv(x, y):
    # lasso importance sampping
    reg = LassoCV(max_iter=10000, normalize=True, tol=1e-3)
    reg.fit(x, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(x, y))
    coef = pd.Series(reg.coef_, index=x.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")
    sort = coef.sort_values()
    print(sort[sort!=0])
#######################################Method 3 #########################################33
# recursive feature elimination, tune to the number of features we want


def recursive_feat_elim(x, y):

    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    sgd = SGDRegressor(max_iter=100000, penalty="elasticnet", alpha=0.00001)
    rfe = RFE(estimator = rf, n_features_to_select=20, step=1)
    x_scaled = scale(x)
    rfe.fit(x_scaled,y)
    ranking = rfe.ranking_.reshape(np.shape(x_scaled)[1])
    # selects top 20 features
    for i, j in enumerate(ranking):
        if j <= 1:
            print(x.columns.values[i])

    #las = Lasso(alpha=0.1, max_iter=1000000)
    #rfe = RFE(estimator=las, n_features_to_select=20, step=1)
    #rfe.fit(x, y)
    #ranking = rfe.ranking_.reshape(np.shape(x)[1])
    #print(ranking)

    #dtr = DecisionTreeRegressor()
    #rfe = RFE(estimator=dtr, n_features_to_select=40, step=1)
    #rfe.fit(x, y)
    #ranking = rfe.ranking_.reshape(np.shape(x)[1])
    #print(ranking)

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
# rfecv

def recursive_feat_cv(x, y):


    sgd = SGDRegressor(max_iter=100000, penalty="elasticnet", alpha=0.00001)
    sgd = SGDRegressor()
    sgd = Lasso(max_iter=100000)
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    svr = SVR(kernel="linear")

    rfecv = RFECV(estimator = svr, min_features_to_select=10, step=1, n_jobs=4, scoring= "explained_variance", verbose = 1)

    rfecv.fit(x,y)
    #ranking = rfe.ranking_.reshape(np.shape(x)[1])
    #print(ranking)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


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
def pca(x):
    pca = PCA(n_components = 20)
    x = scale(x)
    pca.fit(x)
    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3) * 100)
    #print(pca.components_)
    #print(pca.explained_variance_ratio_)
    #print(sum(pca.explained_variance_ratio_))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(0,100)
    plt.style.context('seaborn-whitegrid')
    plt.plot(var)
    plt.show()
#######################################Method 4 #########################################33
#todo: Boruta

def boruta(x,y):
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1,
                             max_iter=500)
    x_scale = scale(x)
    feat_selector.fit(np.array(x_scale), y)
    print(    feat_selector.support_)
    print(feat_selector.ranking_)
    for i, j in enumerate(feat_selector.support_):
        if j == True:
            print(x.columns.values[i])

