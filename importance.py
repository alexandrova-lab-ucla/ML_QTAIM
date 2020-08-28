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


#######################################Method 1 #########################################33
def lasso(x_train, y_train):
    # lasso importance sampping
    scaler = StandardScaler()
    scaler.fit(x_train, y_train)
    print("finished")
    sel = SelectFromModel(Lasso(alpha=0.1, max_iter=10000))
    x_train = scaler.transform(x_train)
    sel.fit(x_train, y_train)
    print(sel.get_support())
    print(np.count_nonzero(sel.get_support()))
    print("finished")


#######################################Method 2 #########################################33
# variance threshold filtering

def variance_thresh(x, y):
    scaler = StandardScaler()
    scaler.fit(x_train, y_train)

    sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
    x_train = scaler.transform(x_train)
    sel.fit(x_train, y_train)


#######################################Method 3 #########################################33
# recursive feature elimination, tune to the number of features we want


def recursive_feat_elim(x, y):
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
