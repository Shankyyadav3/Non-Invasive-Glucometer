import commons, constants, os
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


def build_logistic_regressor(X, Y, kfold, model_folder):
    model = LogisticRegression(max_iter = 10000)
    commons.save_model(model, model_folder + constants.LOG_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_xg_boost_regressor(X, Y, model_folder):  # Mean MAE: 0.330 (0.036)
    model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    commons.save_model(model, model_folder + constants.XGBOOST_REGRESSOR)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model and get scores
    return commons.get_mae_cross_val_score(model, X, Y, cv)


def build_dec_tree_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.298 (0.063)
    model = DecisionTreeRegressor()
    commons.save_model(model, model_folder + constants.DEC_TREE_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_kneigh_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.333 (0.042)
    model = KNeighborsRegressor()
    commons.save_model(model, model_folder + constants.KNEIGH_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_elastic_net_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.362 (0.024)
    model = ElasticNet()
    commons.save_model(model, model_folder + constants.ENET_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_lasso_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.374 (0.023)
    model = Lasso()
    commons.save_model(model, model_folder + constants.LASSO_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_linear_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.337 (0.022)
    model = LinearRegression()
    commons.save_model(model, model_folder + constants.LINEAR_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_ridge_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.337 (0.022)
    model = Ridge()
    commons.save_model(model, model_folder + constants.RIDGE_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def build_SVM_regressor(X, Y, kfold, model_folder):  # Mean MAE: 0.299 (0.042)
    model = SVR()
    commons.save_model(model, model_folder + constants.RIDGE_REGRESSOR)
    return commons.get_mae_cross_val_score(model, X, Y, kfold)


def perform_regression(X, Y, key):
    results_folder = constants.RESULTS+'regression/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    model_folder = constants.MODELS+'regression/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    ind = key.find('_')
    if ind > 0:
        filter_name = key[ind + 1:]
    else:
        filter_name = ''
    if len(filter_name) > 0:
        filter_spec_model_folder = model_folder + filter_name + '/'
    else:
        filter_spec_model_folder = model_folder
    if not os.path.exists(filter_spec_model_folder):
        os.makedirs(filter_spec_model_folder)
    kfold = KFold(n_splits=10)
    results = []
    results.append(constants.REGRESSION_RESULT_COLS)
    results.append([commons.fetch_model_name(constants.XGBOOST_REGRESSOR), filter_name] + build_xg_boost_regressor(X, Y, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.DEC_TREE_REGRESSOR), filter_name] + build_dec_tree_regressor(X, Y, kfold, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.KNEIGH_REGRESSOR), filter_name] + build_kneigh_regressor(X, Y, kfold, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.ENET_REGRESSOR), filter_name] + build_elastic_net_regressor(X, Y, kfold, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.LASSO_REGRESSOR), filter_name] + build_lasso_regressor(X, Y, kfold, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.LINEAR_REGRESSOR), filter_name] + build_linear_regressor(X, Y, kfold, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.RIDGE_REGRESSOR), filter_name] + build_ridge_regressor(X, Y, kfold, filter_spec_model_folder))
    results.append([commons.fetch_model_name(constants.SVM_REGRESSOR), filter_name] + build_SVM_regressor(X, Y, kfold, filter_spec_model_folder))
    return results, results_folder