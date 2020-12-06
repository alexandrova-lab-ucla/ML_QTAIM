import argparse
import multiprocessing as mp

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, SGDRegressor, Ridge, HuberRegressor, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Integer

from extract_helpers_shifted import *
from feature_sel_util import *
from scoring_functions import *

# takes: nothing
# returns: two matrices. list_of_dicts is a list of dictionaries containing
# critical values for each file. Y is the energies of each file.

x, y = extract_all()
min = np.min(y)
max = np.max(y)
# y_scale = (y - min) / (max - min)
std = (np.std(y), np.mean(y))
# std = max - min
y_scale = (y - np.mean(y)) / np.std(y)

pooled_set = \
    [
        "$\mathcal{Bond}_{7}$", "$\mathcal{Bond}_{8}$", "$\mathcal{Bond}_{9}$",
        "$\mathcal{DelocIndBond}_{5}$",
        "$\mathcal{DelSqRho}_{1}$",
        "$\mathcal{DelSqV}_{7}$",
        "$\mathcal{ESP}_{1}$", "$\mathcal{ESP}_{2}$", "$\mathcal{ESP}_{4}$", "$\mathcal{ESP}_{5}$",
        "$\mathcal{ESP}_{6}$",
        "$\mathcal{ESPe}_{10}$",
        "$\mathcal{ESPn}_{4}$", "$\mathcal{ESPn}_{5}$",
        "$\mathcal{HessRhoEigVals}_{c,7}$",
        "$\mathcal{K|Scaled|}_{basic,1}$", "$\mathcal{K|Scaled|}_{basic,2}$",
        "$\mathcal{K|Scaled|}_{basic,3}$", "$\mathcal{K|Scaled|}_{basic,4}$",
        "$\mathcal{K|Scaled|}_{basic,6}$",
        "$\mathcal{Kinetic}_{basic,5}$", "$\mathcal{Kinetic}_{basic,6}$",
        "$\mathcal{Lagr}_{basic,1}$", "$\mathcal{Lagr}_{basic,5}$", "$\mathcal{Lagrangian}_{2}$",
        "$\mathcal{Rho}_{8}$",
        "$\mathcal{Stress_EigVals}_{c,7}$",
        "$\mathcal{Vnuc}_{1}$", "$\mathcal{Vnuc}_{2}$", "$\mathcal{Vnuc}_{3}$",
        "$\mathcal{Vnuc}_{4}$", "$\mathcal{Vnuc}_{5}$", "$\mathcal{Vnuc}_{6}$"
    ]

pool_uncorr = \
    [
        "$\mathcal{Bond}_{7}$", "$\mathcal{Bond}_{8}$", "$\mathcal{Bond}_{9}$",
        "$\mathcal{DelocIndBond}_{5}$",
        "$\mathcal{DelSqRho}_{1}$",
        "$\mathcal{ESP}_{1}$", "$\mathcal{ESP}_{2}$", "$\mathcal{ESP}_{4}$", "$\mathcal{ESP}_{6}$",
        "$\mathcal{ESPn}_{5}$",
        "$\mathcal{HessRhoEigVals}_{c,7}$",
        "$\mathcal{K|Scaled|}_{basic,1}$", "$\mathcal{K|Scaled|}_{basic,2}$",
        "$\mathcal{K|Scaled|}_{basic,3}$", "$\mathcal{K|Scaled|}_{basic,4}$",
        "$\mathcal{Kinetic}_{basic,5}$", "$\mathcal{Kinetic}_{basic,6}$",
        "$\mathcal{Lagr}_{basic,1}$", "$\mathcal{Lagr}_{basic,5}$", "$\mathcal{Lagrangian}_{2}$",
        "$\mathcal{Rho}_{8}$",
        "$\mathcal{Vnuc}_{1}$", "$\mathcal{Vnuc}_{2}$", "$\mathcal{Vnuc}_{3}$",
        "$\mathcal{Vnuc}_{4}$", "$\mathcal{Vnuc}_{5}$", "$\mathcal{Vnuc}_{6}$"
    ]

# physical set, general model
physical = \
    [
        "$\mathcal{Bond}_{7}$", "$\mathcal{Bond}_{8}$", "$\mathcal{Bond}_{9}$", "$\mathcal{Bond}_{10}$",
        "$\mathcal{DelocIndBond}_{5}$",
        "$\mathcal{ESP}_{1}$", "$\mathcal{ESP}_{2}$", "$\mathcal{ESP}_{3}$", "$\mathcal{ESP}_{4}$",
        "$\mathcal{ESP}_{5}$", "$\mathcal{ESP}_{6}$",
        "$\mathcal{ESPn}_{4}$", "$\mathcal{ESPn}_{5}$",
        "$\mathcal{HessRhoEigVals}_{c,7}$",
        "$\mathcal{K|Scaled|}_{basic,1}$", "$\mathcal{K|Scaled|}_{basic,2}$", "$\mathcal{K|Scaled|}_{basic,3}$",
        "$\mathcal{K|Scaled|}_{basic,4}$", "$\mathcal{K|Scaled|}_{basic,5}$", "$\mathcal{K|Scaled|}_{basic,6}$",
        "$\mathcal{Vnuc}_{1}$", "$\mathcal{Vnuc}_{2}$", "$\mathcal{Vnuc}_{3}$", "$\mathcal{Vnuc}_{4}$",
        "$\mathcal{Vnuc}_{5}$",
        "$\mathcal{Vnuc}_{6}$"
    ]

parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
parser.add_argument("--algo", action='store', dest="algo", default="xgb", help="select algorithm")
parser.add_argument("-n", action='store', dest="n_iter", default="500", help="select number of trials")
parser.add_argument('--bayes', dest="bayes", action='store_true')
parser.add_argument('--single', dest="single", action='store_true')
parser.add_argument('--pool', dest="pool", action='store_true')
parser.add_argument('--physical', dest="phys", action='store_true')
parser.add_argument('--all', dest="all", action='store_true')

results = parser.parse_args()
algo = results.algo
bayes = results.bayes
single = results.single
n_iter = int(results.n_iter)
all = results.all
pool = results.pool
physical_set = results.phys

phys_x_df = x[physical]
pool_x_uncorr_df = x[pool_uncorr]
pool_x_df = x[pooled_set]

full_input = scale(x)
pool_x = scale(x[pooled_set].to_numpy())
phys_x = scale(x[physical].to_numpy())
pool_x_uncorr = scale(x[pool_uncorr].to_numpy())

# selection of what feature set to use
if (pool == True):
    dataset = pool_x
    ref_df = pool_x_df
else:
    if (physical_set == True):
        dataset = phys_x
        ref_df = phys_x_df
    else:
        if (all == True):
            dataset = full_input
            ref_df = x
        else:
            dataset = pool_x_uncorr
            ref_df = pool_x_uncorr_df

x_train, x_test, y_train, y_test = train_test_split(ref_df, y_scale, test_size=0.2, random_state=1)
names = ref_df

if (bayes == True):
    # dictionaries of hyperparameters
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
    params_rf = {"min_samples_split": Integer(2, 8),
                 "n_estimators": Integer(100, 1000),
                 "min_samples_leaf": Integer(1, 10),
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
        "learning_rate": Real(0.01, 0.25),
        "alpha": Real(0, 0.2),
        "eta": Real(0, 0.2),
        "gamma": Real(0, 0.2),
        "n_estimators": Integer(100, 2000),
        "objective": ["reg:squarederror"],
        "tree_method": ["gpu_hist"]}
    param_grad = {"n_estimators": Integer(100, 2000),
                  "learning_rate": Real(1e-5, 0.1, prior="log-uniform"),
                  "subsample": Real(0.8, 1),
                  "min_samples_split": Integer(2, 5),
                  "min_samples_leaf": Integer(1, 4),
                  "max_depth": Integer(3, 15)}
    params_ridge = {"tol": Real(1e-5, 1e-1, prior="log-uniform"), "alpha": Real(1e-2, 10, prior="log-uniform")}
    params_gp = {"alpha": Real(1e-12, 1e-4, prior="log-uniform")}
    params_lasso = {"alpha": Real(1e-5, 1, prior="log-uniform")}
    param_ada = {"n_estimators": Integer(1e1, 1e3, prior="log-uniform"),
                 "learning_rate": Real(1e-2, 1e1, prior="log-uniform")}
    param_extra = {"n_estimators": Integer(10, 1e4, prior="log-uniform"),
                   "min_samples_split": Integer(2, 6), "min_samples_leaf": Integer(2, 4),
                   "max_depth": Integer(10, 50), "n_jobs": [mp.cpu_count()]}
    param_huber = {"epsilon": Real(1.01, 1.5), "alpha": Real(1e-6, 1e-1, prior="log-uniform"),
                   "tol": Real(1e-7, 1e-2, prior="log-uniform")}
    param_knn = {"n_neighbors": Integer(3, 7)}

    if (algo == "xgb"):
        print("xgb algorithms")
        reg_xgb = xgb.XGBRegressor()
        reg_xgb = BayesSearchCV(reg_xgb, params_xgb, n_iter=n_iter, verbose=4, cv=3)
        custom_scorer_xgb = custom_skopt_xgb_scorer
        reg_xgb.fit(x_train, y_train, callback=[custom_skopt_xgb_scorer(x, y)])  # custom scoring functions
        score(reg_xgb, x_train, x_test, y_train, y_test, std)

    elif (algo == "svr_rbf"):
        print("svr rbf algorithms")
        reg_svr_rbf = SVR(kernel="rbf")
        reg_svr_rbf = BayesSearchCV(reg_svr_rbf, params_svr_rbf, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_svr_rbf.fit(x_train, y_train)
        score(reg_svr_rbf, x_train, x_test, y_train, y_test, std)

    elif (algo == "svr_lin"):
        print("svr lin algorithms")
        reg_svr_lin = SVR(kernel="linear")
        reg_svr_lin = BayesSearchCV(reg_svr_lin, params_svr_lin, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_svr_lin.fit(x_train, y_train)
        score(reg_svr_lin, x_train, x_test, y_train, y_test, std)

    elif (algo == "bayes"):
        print("bayes algorithm")
        reg_bayes = BayesianRidge()
        reg_bayes = BayesSearchCV(reg_bayes, params_bayes, n_iter=n_iter, verbose=3, cv=3, n_jobs=10,
                                  scoring="neg_mean_absolute_error")
        reg_bayes.fit(x_train, y_train)
        score(reg_bayes, x_train, x_test, y_train, y_test, std)

    elif (algo == "rf"):
        print("random forest algorithms ")
        reg_rf = RandomForestRegressor()
        reg_rf = BayesSearchCV(reg_rf, params_rf, n_iter=n_iter, verbose=3, cv=3, n_jobs=10,
                               scoring="neg_mean_absolute_error")
        custom_scorer_rf = custom_skopt_rf_scorer
        reg_rf.fit(x_train, y_train, callback=[custom_skopt_rf_scorer(x, y)])  # custom scoring functions
        score(reg_rf, x_train, x_test, y_train, y_test, std)

    elif (algo == "sgd"):
        print("sgd algorithms")
        reg_sgd = SGDRegressor()
        reg_sgd = BayesSearchCV(reg_sgd, params, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_sgd.fit(x_train, y_train)
        score(reg_sgd, x_train, x_test, y_train, y_test, std)

    elif (algo == "lasso"):
        print("lasso algorithms")
        reg_lasso = Lasso()
        reg_lasso = BayesSearchCV(reg_lasso, params_lasso, n_iter=n_iter, cv=3)
        reg_lasso.fit(x_train, y_train)
        score(reg_lasso, x_train, x_test, y_train, y_test, std)

    elif (algo == "ridge"):
        print("ridge algorithms")
        reg_ridge = Ridge()
        reg_ridge = BayesSearchCV(reg_ridge, params_ridge, verbose=4, n_iter=n_iter, cv=3)
        reg_ridge.fit(x_train, y_train)
        score(reg_ridge, x_train, x_test, y_train, y_test, std)

    elif (algo == "gp"):
        print("gaussian process algorithm")
        kernel = Matern(length_scale=1, nu=5 / 2) + WhiteKernel(noise_level=1)
        reg_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        reg_gp = BayesSearchCV(reg_gp, params_gp, n_iter=n_iter, verbose=4, cv=5)
        reg_gp.fit(x_train, y_train)
        score(reg_gp, x_train, x_test, y_train, y_test, std)  # fuck with kernel

    elif (algo == "krr"):
        print("krr algorithm")
        reg_kernelridge = KernelRidge(kernel="poly", degree=8)
        reg_kernelridge = BayesSearchCV(reg_kernelridge, params_kernelridge, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_kernelridge.fit(x_train, y_train)
        score(reg_kernelridge, x_train, x_test, y_train, y_test, std)

    elif (algo == "ada"):
        print("ada algorithm")

        reg_ada = AdaBoostRegressor()
        reg_ada = BayesSearchCV(reg_ada, param_ada, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_ada.fit(x_train, y_train)
        score(reg_ada, x_train, x_test, y_train, y_test, std)

    elif (algo == "nn"):
        print("nn algorithm")

        reg_nn = MLPRegressor(early_stopping=True, n_iter_no_change=n_iter, hidden_layer_sizes=(200, 200,),
                              solver="adam")
        reg_nn = BayesSearchCV(reg_nn, params_nn, n_iter=n_iter, verbose=3, cv=3, n_jobs=10,
                               scoring="neg_mean_absolute_error")
        reg_nn.fit(x_train, y_train)
        score(reg_nn, x_train, x_test, y_train, y_test, std)

    elif (algo == "extra"):
        print("extra algorithm")

        reg_extra = ExtraTreesRegressor(criterion="mae")
        reg_extra = BayesSearchCV(reg_extra, param_extra, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        custom_scorer_extra = custom_skopt_extra_scorer
        reg_extra.fit(x_train, y_train, callback=[custom_scorer_extra(x, y)])  # custom scoring functions
        score(reg_extra, x_train, x_test, y_train, y_test, std)

    elif (algo == "huber"):
        print("huber algorithm")

        reg_huber = HuberRegressor(max_iter=1000)
        reg_huber = BayesSearchCV(reg_huber, param_huber, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_huber.fit(x_train, y_train)
        score(reg_huber, x_train, x_test, y_train, y_test, std)

    elif (algo == "knn"):
        print("knn algorithm")

        reg_knn = KNeighborsRegressor(algorithm="auto", weights="distance")
        reg_knn = BayesSearchCV(reg_knn, param_knn, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_knn.fit(x_train, y_train)
        score(reg_knn, x_train, x_test, y_train, y_test, std)

    elif (algo == "grad"):
        print("grad algorithm")
        reg_grad = GradientBoostingRegressor(n_iter_no_change=250)
        reg_grad = BayesSearchCV(reg_grad, param_grad, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        reg_grad.fit(x_train, y_train, callback=[custom_skopt_grad_scorer(x, y)])  # custom scoring functions
        score(reg_grad, x_train, x_test, y_train, y_test, std)

    else:
        print("extra trees algorithm")
        reg_extra = ExtraTreesRegressor(criterion="mae")
        reg_extra = BayesSearchCV(reg_extra, param_extra, n_iter=n_iter, verbose=3, cv=3, n_jobs=10)
        custom_scorer_extra = custom_skopt_extra_scorer
        reg_extra.fit(x_train, y_train, callback=[custom_scorer_extra(x, y)])
        score(reg_extra, x_train, x_test, y_train, y_test, std)

elif (single == True):

    if (algo == "svr_rbf"):
        print("svr rbf algorithms")
        reg_svr_rbf = SVR(kernel="rbf", C=0.6299017591106881, cache_size=500,
                          epsilon=0.056183687320042426, gamma=0.059982132068042655)
        reg_svr_rbf.fit(x_train, y_train)
        score_single(reg_svr_rbf, x_train, x_test, y_train, y_test, std)

    elif (algo == "svr_lin"):
        print("svr lin algorithms")
        reg_svr_lin = SVR(kernel="linear")
        reg_svr_lin.fit(x_train, y_train)
        score_single(reg_svr_lin, x_train, x_test, y_train, y_test, std)

    elif (algo == "bayes"):
        print("bayes algorithm")
        reg_bayes = BayesianRidge()
        reg_bayes.fit(x_train, y_train)
        score_single(reg_bayes, x_train, x_test, y_train, y_test, std)

    elif (algo == "sgd"):
        print("sgd algorithms")
        reg_sgd = SGDRegressor()
        reg_sgd.fit(x_train, y_train)
        score_single(reg_sgd, x_train, x_test, y_train, y_test, std)

    elif (algo == "lasso"):
        print("lasso algorithms")
        reg_lasso = Lasso(alpha=0.01)
        reg_lasso.fit(x_train, y_train)
        score_single(reg_lasso, x_train, x_test, y_train, y_test, std)

    elif (algo == "ridge"):
        print("ridge algorithms")
        reg_ridge = Ridge(alpha=10.0, tol=1e-05)
        reg_ridge.fit(x_train, y_train)
        score_single(reg_ridge, x_train, x_test, y_train, y_test, std)

    elif (algo == "gp"):

        import gpflow
        from gpflow.ci_utils import ci_niter
        from tensorflow_probability import distributions as tfd
        import tensorflow_probability as tfp
        import tensorflow as tf

        opt = gpflow.optimizers.Scipy()
        k = gpflow.kernels.RationalQuadratic()
        k = gpflow.kernels.Matern52()
        # k = Tanimoto()

        reg_gp = gpflow.models.GPR(data=(x_train, y_train.reshape(-1, 1)), kernel=k,
                                   noise_variance=1)
        opt_logs = opt.minimize(reg_gp.training_loss, reg_gp.trainable_variables, options=dict(maxiter=10000), )
        f64 = gpflow.utilities.to_default_float

        # option for two different, common kernels from cheminformatics

        # matern
        reg_gp.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
        reg_gp.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        reg_gp.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

        # tanimoto
        # reg_gp.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
        # reg_gp.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

        num_burnin_steps = ci_niter(500)
        num_samples = ci_niter(1000)

        # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
        hmc_helper = gpflow.optimizers.SamplingHelper(
            reg_gp.log_posterior_density, reg_gp.trainable_parameters
        )

        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
        )
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
        )


        @tf.function
        def run_chain_fn():
            return tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=hmc_helper.current_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )


        samples, _ = run_chain_fn()
        parameter_samples = hmc_helper.convert_to_constrained_values(samples)
        param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(reg_gp).items()}

        # gp returns mean and standard deviation of sample dist.
        y_pred, y_var = reg_gp.predict_f(np.array(x_test))
        y_pred_test, y_var = reg_gp.predict_f(np.array(x_test))
        y_pred_train, y_var = reg_gp.predict_f(np.array(x_train))

        print(reg_gp.training_loss)

        print("------------------------------")
        print("gaussian process algorithm")
        # statistics
        mse_test = str(mean_squared_error(y_test * std[0], y_pred_test * std[0]))
        mse_train = str(mean_squared_error(y_train * std[0], y_pred_train * std[0]))
        mae_test = str(mean_absolute_error(y_test * std[0], y_pred_test * std[0]))
        mae_train = str(mean_absolute_error(y_train * std[0], y_pred_train * std[0]))
        r2_test = str(r2_score(y_test, y_pred_test))
        r2_train = str(r2_score(y_train, y_pred_train))

        print("MSE test score: \t" + str(mse_test))
        print("MSE train score:\t" + str(mse_train))
        print("MAE test score: \t" + str(mae_test))
        print("MAE train score:\t" + str(mae_train))
        print("r2 score test: \t\t" + str(r2_test))
        print("r2 score train:\t\t" + str(r2_train))

    elif (algo == "krr"):
        print("krr algorithm")
        reg_kernelridge = KernelRidge(kernel="poly", degree=8)
        reg_kernelridge.fit(x_train, y_train)
        score_single(reg_kernelridge, x_train, x_test, y_train, y_test, std)

    elif (algo == "ada"):
        print("ada algorithm")

        reg_ada = AdaBoostRegressor()
        reg_ada.fit(x_train, y_train)
        score_single(reg_ada, x_train, x_test, y_train, y_test, std)

    elif (algo == "nn"):

        print("nn algorithm")

        reg_nn = MLPRegressor(early_stopping=True, n_iter_no_change=n_iter,
                              hidden_layer_sizes=(200, 200,),
                              solver="lbfgs", alpha=8.64e-10,
                              learning_rate_init=3.84e-05,
                              max_iter=640, tol=1.86e-03)
        reg_nn.fit(x_train, y_train)
        score_single(reg_nn, x_train, x_test, y_train, y_test, std)

        # tensorflow
        import tensorflow as tf
        from tensorflow.keras import regularizers

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu', input_shape=(np.shape(x_train)[1],),
                                  bias_regularizer=regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="linear")
        ])

        model.compile(optimizer='adam',
                      loss="MSE",
                      metrics=["MAE", "MSE"])
        model.fit(x_train, np.ravel(y_train), epochs=2000, batch_size=np.shape(x_train)[0], verbose=1)

        y_hat = model.predict(x_test)
        mse = mean_squared_error(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)
        print("------------------------------")
        print(np.shape(x_train))
        print("mae test:" + str(mae * (std[0])))
        print("mse test:" + str(mse))
        print("r2 test:" + str(r2))
        # tensorflow

    elif (algo == "xgb"):

        print("xgb algorithms")
        reg_xgb = xgb.XGBRegressor(
            reg_alpha=0.01, colsample_bytree=0.3, eta=0.0, gamma=0.0,
            reg_lambda=0.0, learning_rate=0.01, max_depth=5, n_estimators=1000,
            objective="reg:squarederror", tree_method="gpu_hist")
        reg_xgb.fit(x_train, y_train)
        score_single(reg_xgb, x_train, x_test, y_train, y_test, std)

    elif (algo == "rf"):
        print("random forest algorithms ")
        reg_rf = RandomForestRegressor(min_samples_leaf=1, min_samples_split=2,
                                       n_estimators=2500, n_jobs=10, criterion="entropy")
        reg_rf = RandomForestRegressor(n_jobs=10)
        custom_scorer_rf = custom_skopt_rf_scorer
        reg_rf.fit(x_train, y_train)
        score_single(reg_rf, x_train, x_test, y_train, y_test, std)

    elif (algo == "extra"):
        print("extra algorithm")

        reg_extra = ExtraTreesRegressor(min_samples_split=3,
                                        min_samples_leaf=3,
                                        n_estimators=2200)
        custom_scorer_extra = custom_skopt_extra_scorer
        reg_extra.fit(x_train, y_train)
        score_single(reg_extra, x_train, x_test, y_train, y_test, std)

    elif (algo == "grad"):
        print("grad algorithm")
        dict = {'learning_rate': 0.029850214667088548, 'min_samples_split': 3, 'min_samples_leaf': 4, 'max_depth': 2,
                'n_estimators': 611, 'subsample': 0.4}
        reg_grad = GradientBoostingRegressor(**dict)
        reg_grad.fit(x_train, y_train)
        score_single(reg_grad, x_train, x_test, y_train, y_test, std)

    elif (algo == "huber"):
        print("huber algorithm")
        reg_huber = HuberRegressor(max_iter=1000, alpha=1e-06, epsilon=1.01, tol=0.0012904985834892478)
        reg_huber.fit(x_train, y_train)
        score_single(reg_huber, x_train, x_test, y_train, y_test, std)

    elif (algo == "knn"):
        print("knn algorithm")
        reg_knn = KNeighborsRegressor(algorithm="auto", weights="distance", n_neighbors=5)
        reg_knn.fit(x_train, y_train)
        score_single(reg_knn, x_train, x_test, y_train, y_test, std)

    else:
        print("extra trees algorithm")
        reg_extra = ExtraTreesRegressor(min_samples_split=3,
                                        min_samples_leaf=3,
                                        n_estimators=2200)
        reg_extra.fit(x_train, y_train)
        score_single(reg_extra, x_train, x_test, y_train, y_test, std)

else:
    print("no training selected, feature selection")
    #pca(x, list(x), y)
    # lasso(x,y)
    # lasso_cv(x,y)
    # 15 pca components has 82% explained variance
    # 20 pca components has 87% explained variance
    # 25 pca components has 90% explained variance
    # lasso(x, y)
    # lasso_cv(x, y)

    #boruta(x,y, n = 7)
    #boruta(x,y, n = 5)
    #boruta(x,y, n = 3)

    print("dendrogram")
    dendo(names)
    print("quantitative feature selction")
    quant_feat(x_train, x_test, y_train, y_test, names)

