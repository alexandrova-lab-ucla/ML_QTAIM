import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from boruta import BorutaPy
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV, \
    SelectFromModel, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor

#######################################Method 1: Lasso #########################################33
def lasso(x, y):
    # lasso importance sampping
    #x_scaled = scale(x)

    print("passed scale")
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x, y)

    sel = SelectFromModel(Lasso(alpha=0.021324, max_iter=50000, normalize = True))
    sel.fit(x_scaled, y)

    print("number of features selected via lasso: " + str(np.count_nonzero(sel.get_support())))
    for i, j in enumerate(sel.get_support()):
        if j != 0:
            print(x.columns.values[i])

def lasso_cv(x, y):
    # lasso importance sampping
    reg = LassoCV( alphas = np.logspace(-3, -1, num = 150), normalize = True, cv = 5 )
    reg.fit(x, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(x, y))
    coef = pd.Series(reg.coef_, index=x.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
        sum(coef == 0)) + " variables")
    sort = coef.sort_values()
    print(sort[sort!=0])
#######################################Method 2: Recursive #########################################33
# recursive feature elimination, tune to the number of features we want


def recursive_feat_elim(x, y):

    rf = RandomForestRegressor(n_jobs=-1, max_depth=3)
    sgd = SGDRegressor(max_iter=100000, penalty="elasticnet", alpha=0.00001)
    svr = SVR(kernel="linear")

    rfe = RFE(estimator = rf, n_features_to_select = 5, step=1, verbose=1)

    rfe.fit(x,y)
    ranking = rfe.ranking_.reshape(np.shape(x)[1])
    #selects top 20 features
    for i, j in enumerate(ranking):
        if j <= 1:
            print(x.columns.values[i])


#######################################Method 3: Recursive  #########################################33

def recursive_feat_cv(x, y):


    sgd = SGDRegressor(max_iter=100000, penalty="elasticnet", alpha=0.00001)
    sgd = SGDRegressor()
    sgd = Lasso(max_iter=100000)
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    #svr = SVR(kernel="linear")
    rf = RandomForestRegressor(n_jobs=-1, max_depth=7)
    rfecv = RFECV(estimator = rf, min_features_to_select=10, step=1, n_jobs=4, scoring= "explained_variance", verbose = 1)
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


#######################################Method 4: Var Threshold #########################################33
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

#######################################Method 5: PCA #########################################33
def pca(x, labels=[], barriers=np.array([])):


    pca = PCA(n_components=3)
    x = StandardScaler().fit_transform(x)
    pca.fit(x)
    result = pd.DataFrame(pca.transform(x), columns=['PCA%i' % i for i in range(3)])
    barriers = np.delete(barriers, np.argmax(result['PCA0']), 0)
    pc_0 = result['PCA0'].to_numpy()
    pc_1 = result['PCA1'].to_numpy()
    pc_2 = result['PCA2'].to_numpy()
    pc_0 = np.delete(pc_0, np.argmax(result['PCA0']), 0)
    pc_1 = np.delete(pc_1, np.argmax(result['PCA0']), 0)
    pc_2 = np.delete(pc_2, np.argmax(result['PCA0']), 0)

    min = np.min(barriers)
    max = np.max(barriers)
    mid = (min + max)/2
    norm = cm.colors.Normalize(vmin=np.min(barriers), vmax=np.max(barriers))

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pl = ax.scatter(pc_0, pc_1, pc_2,
                    s=10, cmap="bwr",
                    c=[norm(item) for item in barriers],
                    alpha=0.5)
    ax.set_xlabel('PCA0')
    ax.set_ylabel('PCA1')
    ax.set_zlabel('PCA2')
    cbar = fig.colorbar(pl, cmap="bwr", ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_label("Barrier Energy")
    cbar.ax.set_yticklabels([str(min), str((min + mid)/2), str(mid), str((max+mid)/2), str(max)])
    plt.show()
    '''
    comp = []
    coeff_pca = []
    for i in pca.components_:
        for ind, j in enumerate(labels):
            if(np.absolute(i[ind]) > 0.1):
                comp.append(labels[ind])
                coeff_pca.append(i[ind])
        print(comp)
        print(len(comp))

        #print(np.sort(np.absolute(np.array(i))))
        #print(len(np.array(i)))
        # top n features
        #i = np.array(i)
        #ind = np.argsort(np.absolute(np.array(i))).tolist()
        #print(i[ind[-20:-1]])
        #for ind_temp in ind:
        #    print(i[ind_temp])
        ind = []
        comp = []

    #print(pca.components_)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    '''
    fig, axs = plt.subplots(2,2)
    axs[0,0].scatter(pc_1, pc_2,
                    s=10, cmap="seismic",
                    c=[norm(item) for item in barriers],
                    alpha=0.3)
    axs[1,0].scatter(pc_0, pc_2,
                    s=10, cmap="seismic",
                    c=[norm(item) for item in barriers],
                    alpha=0.3)
    tmp = axs[0,1].scatter(pc_1, pc_0,
                    s=10, cmap="seismic",
                    c=[norm(item) for item in barriers],
                    alpha=0.3)

    axs[0,0].set(xlabel="PC1", ylabel='PC2')
    axs[1,0].set(xlabel="PC0", ylabel='PC2')
    axs[0,1].set(xlabel="PC1", ylabel='PC0')

    #fig.subplots_adjust(right=0.8)
    cbaxes = fig.add_axes([0.90, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(tmp, cmap="seismic", ticks=[0, 0.25, 0.5, 0.75, 1],
                        ax = axs[:,1], cax=cbaxes)

    cbar.set_label("Barrier Energy")
    cbar.ax.set_yticklabels([str(int(min)), str(int((min + mid) / 2)), str(int(mid)), str(int((max + mid) / 2)), str(int(max))])

    plt.tight_layout(rect=[0,0,0.9,1])
    fig.suptitle("2D Principal Components")
    fig.subplots_adjust(top=0.88)
    plt.show()
    '''

    '''
    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3) * 100)
    seaborn.set_theme(style="ticks")
    fig, ax = plt.subplots()
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('Explained Variability vs. # Eigenvectors')
    plt.ylim(30,100)
    plt.xlim(0,50)
    plt.style.context('seaborn-whitegrid')
    #plt.plot(var)
    seaborn.lineplot(range(len(var)), var)
    plt.hlines(95, linestyles= "dashdot", xmin=0, xmax=50)
    plt.hlines(90, linestyles= "dashdot", xmin=0, xmax=50)
    plt.hlines(85, linestyles= "dashdot", xmin=0, xmax=50)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(33, 80, "95% Variance Explained", fontsize=9,
            verticalalignment='top', bbox=props)
    ax.text(24, 75, "90% Variance Explained", fontsize=9,
            verticalalignment='top', bbox=props)
    ax.text(14, 70, "85% Variance Explained", fontsize=9,
            verticalalignment='top', bbox=props)
    ax.arrow(21,71, 0, 14, fc='black',             #arrow fill color
             ec='black')
    ax.arrow(30,76, 0, 14,fc='black',             #arrow fill color
             ec='black')
    ax.arrow(40,81, 0, 14, fc='black',             #arrow fill color
             ec='black')
    plt.show()
    '''
#######################################Method 6: Boruta #########################################33

def boruta(x,y, n=5):
    x_scale = scale(x)

    #rf = RandomForestRegressor(n_jobs=-1, max_depth=n)
    #rf.fit(x_scale, y)
    #y_test = rf.predict(x_scale)
    #mse = mean_squared_error(y_test, y)
    #mae = mean_absolute_error(y_test, y)
    #r2 = r2_score(y_test, y)
    #print("r2: " + str(r2))
    #print("mse: "+ str(mse))
    #print("mae: "+ str(mae))
    rf = RandomForestRegressor(n_jobs=-1, max_depth=n)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, max_iter=2500)
    #for i in x:
    #    print(i)
    feat_selector.fit(np.array(x_scale), y)
    #print(    feat_selector.support_)
    #print(feat_selector.ranking_)
    for i, j in enumerate(feat_selector.support_):
        if j == True:
            print(x.columns.values[i])

