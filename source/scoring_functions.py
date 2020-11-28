from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor,\
    GradientBoostingRegressor
import xgboost as xgb
import numpy as np
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
        reg = xgb.XGBRegressor(**dict)

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

