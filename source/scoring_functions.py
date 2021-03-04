import numpy as np
import xgboost as xgb
import seaborn as sns
sns.set_style("ticks")
sns.set_context("paper")
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,\
    GradientBoostingRegressor

matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': 'cmr10',
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            'font.size': 16,
                            })

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
        if (mae < 11):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(param_extra)
        return 0

class custom_skopt_grad_scorer(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, res):
        param_grad = {"learning_rate": res["x"][0],
                      "min_samples_split": res["x"][1],
                      "min_samples_leaf": res["x"][2],
                      "max_depth": res["x"][3],
                      "n_estimators": res["x"][4],
                      "subsample": res["x"][5]}

        reg = GradientBoostingRegressor(**param_grad, criterion="mse")
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        reg.fit(x_train, y_train)
        y_pred = np.array(reg.predict(x_test))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if (mae < 12):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(param_grad)
        return 0

class custom_skopt_xgb_scorer(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, res):

        param_xgb = {
            "objective":"reg:squarederror", "tree_method":"gpu_hist",
             "alpha":res["x"][0], "colsample_bytree":res["x"][1],"eta":res["x"][2],
            "gamma":res["x"][3], "lambda": res["x"][4], "learning_rate":res["x"][5],
            "max_depth":res["x"][6],"n_estimators":res["x"][7]
        }
        reg = xgb.XGBRegressor(**param_xgb)

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        reg.fit(x_train, y_train)
        y_pred = np.array(reg.predict(x_test))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if (mae < 11):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(param_xgb)
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
        if (mae < 12):
            print("------------------------------")
            print("mae test:" + str(mae))
            print("mse test:" + str(mse))
            print("r2 test:" + str(r2))
            print(params_rf)
        return 0

def score_single(reg, x_train, x_test, y_train, y_test, scale=(1,0)):

    print("................................................")
    try:
        score = reg.score(list(x_test), y_test)
        score_train = reg.score(list(x_train), y_train)

    except:
        score = reg.score(x_test, y_test)
        score_train = reg.score(x_train, y_train)

    #print("score:     \t\t" + str(score))
    #print("score train\t\t" + str(score_train))
    y_pred_train = reg.predict(x_train)
    y_pred_test  = reg.predict(x_test)

    mse_test = str(mean_squared_error(y_test * scale[0], y_pred_test * scale[0]))
    mse_train = str(mean_squared_error(y_train * scale[0], y_pred_train * scale[0]))
    mae_test = str(mean_absolute_error(y_test * scale[0], y_pred_test * scale[0]))
    mae_train = str(mean_absolute_error(y_train * scale[0], y_pred_train * scale[0]))
    r2_test = str(r2_score(y_test, y_pred_test))
    r2_train = str(r2_score(y_train, y_pred_train))

    #print("MSE test score: \t" + str(mse_test))
    #print("MSE train score:\t" + str(mse_train))
    print("----------------------------------------------------")
    print("MAE test score: \t" + str(mae_test))
    print("MAE train score:\t" + str(mae_train))
    print("----------------------------------------------------")
    print("r2 score test: \t\t" + str(r2_test))
    print("r2 score train:\t\t" + str(r2_train))

    plt.plot(y_pred_train*scale[0]+ scale[1],y_train*scale[0] + scale[1], 'o', color='blue', markersize = 5, label= "Train")
    plt.plot(y_pred_test*scale[0]+ scale[1],y_test*scale[0]+ scale[1], 'o', color='red', label = "Test")
    plt.legend(fontsize = 12)
    x = np.linspace(0,300,100)
    plt.plot(x, x, color="black")
    plt.ylim((0,300))
    plt.xlim((0, 300))

    plt.ylabel("True Value [kJ/mol]", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel("Predicted Value [kJ/mol]", fontsize=16)
    plt.xticks(fontsize=12)
    name = str(reg)
    plt.title("Ridge Parity", fontsize=16)
    plt.show()
    plt.tight_layout()

    resid = [np.abs(y_test[i] - y_pred_test[i]) for i in range(len(y_test))]
    sorted = np.argsort(resid) # decreasing order
    [worst1, worst2, worst3] = [i for i in sorted[len(sorted)-3:len(sorted)]]

    print("----------------------------------------------------")
    print("worst residuals: "+ str(resid[worst1])+ ", "+ str(resid[worst2])+", " + str(resid[worst3]))
    print("worst index: " +str(worst1)+ ", "+str(worst2)+ ", " +str(worst3))
    print("1st worst barrier prediction: " +str(y_test[worst1]*scale[0]+ scale[1]))
    print("2st worst barrier prediction: " +str(y_test[worst2]*scale[0]+ scale[1]))
    print("3st worst barrier prediction: " +str(y_test[worst3]*scale[0]+ scale[1]))
    print("1st worst resid: " +str((y_test[worst1] - y_pred_test[worst1])*scale[0]))
    print("2st worst resid: " +str((y_test[worst2] - y_pred_test[worst2])*scale[0]))
    print("3st worst resid: " +str((y_test[worst3] - y_pred_test[worst3])*scale[0]))
    print(scale)
    #plt.show()

    x_test_sans = x_test.drop([x_test.index[worst1], x_test.index[worst2], x_test.index[worst3]])
    y_test_sans = np.delete(y_test, [worst1, worst2, worst3])
    y_pred_test  = reg.predict(x_test_sans)
    mse_test = str(mean_squared_error(y_test_sans * scale[0], y_pred_test * scale[0]))
    mae_test = str(mean_absolute_error(y_test_sans * scale[0], y_pred_test * scale[0]))
    r2_test = str(r2_score(y_test_sans, y_pred_test))
    print("----------------------------------------------------")
    print("MSE test score: \t" + str(mse_test))
    print("MAE test score: \t" + str(mae_test))
    print("r2 score test: \t\t" + str(r2_test))
    print("----------------------------------------------------")

    plt.clf()
    print("Std. Var: " + str(scale[0]) + " Mean: " + str(scale[1]))
    sns.boxplot(np.array(resid) * scale[0])
    plt.title("Residual Distribution, Grad. Boost", fontsize = 18)
    plt.xlabel("Abs. Residual Error [kJ/mol]", fontsize = 16)
    plt.show()
    #plt.clf()
    #sns.histplot(resid * scale)
    #plt.show()

def score(reg, x_train, x_test, y_train, y_test, scale=1):

    print("................................................")
    try:
        score = reg.score(list(x_test), y_test)
    except:
        score = reg.score(x_test, y_test)

    print("score:                " + str(score))
    score = str(mean_squared_error(reg.predict(x_test) * scale[0], y_test * scale[0]))
    print("MSE score test:   " + str(score))
    mae_test = str(mean_absolute_error(reg.predict(x_test) * scale[0], y_test * scale[0]))
    print("MAE score test:   " + str(mae_test))
    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score test:   " + str(score))

    score = str(mean_squared_error(reg.predict(x_train) * scale[0], y_train * scale[0]))
    print("MSE train score:   " + str(score))
    mae_train = str(mean_absolute_error(reg.predict(x_train) * scale[0], y_train * scale[0]))
    print("MAE train score:   " + str(mae_train))
    score = str(r2_score(reg.predict(x_train), y_train))
    print("r2 score train:   " + str(score))
    print(reg.best_estimator_)

    plt.plot(y_train, reg.predict(x_train), 'o', color='black', markersize = 5)
    plt.plot(y_test, reg.predict(x_test), 'o', color='red')
    #plt.text(0.5, 0.15, "MAE test: " +str(mae_test))
    #plt.text(0.5, 0.25, "MAE train: " +str(mae_train))

    plt.xlabel("Normalized Predicted Value", fontsize=16)
    plt.ylabel("Normalized True Values", fontsize=16)


    name = str(reg.get_params()["estimator"]).split("(")[0]
    plt.title(name, fontsize=16)
    plt.savefig(name + ".png")
    plt.clf()
