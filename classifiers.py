import commons, constants, os
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier, LogisticRegressionCV #, LinearClassifier
from sklearn.neural_network import MLPClassifier

def perform_classification(X, Y, key):
    results_folder = constants.RESULTS + 'classification/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    model_folder = constants.MODELS + 'classification/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    ind = key.find('_')
    if ind > 0:
        filter_name = key[ind+1:]
    else:
        filter_name = ''
    if len(filter_name) > 0:
        filter_spec_model_folder = model_folder + filter_name + '/'
    else:
        filter_spec_model_folder = model_folder
    if not os.path.exists(filter_spec_model_folder):
        os.makedirs(filter_spec_model_folder)
    X_train, X_test, y_train, y_test = commons.get_train_test_split(X, Y)

    results = []
    results.append(constants.CLASSIFICATION_RESULT_COLS)
    results.append([commons.fetch_model_name(constants.DEC_TREE_CLASSIFIER), filter_name] + build_decision_tree_classifier(X_train, X_test,
                                                                                                  y_train, y_test,
                                                                                                  filter_spec_model_folder))
    results.append(
        [commons.fetch_model_name(constants.RAND_FOREST_CLASSIFIER), filter_name] + build_random_forest_classifier(X_train, X_test,
                                                                                                   y_train, y_test,
                                                                                                   filter_spec_model_folder))
    results.append(
        [commons.fetch_model_name(constants.KNEIGH_CLASSIFIER), filter_name] + build_KNN_classifier(X_train, X_test,
                                                                                                      y_train, y_test,
                                                                                                      filter_spec_model_folder))
    results.append(
        [commons.fetch_model_name(constants.RIDGE_CLASSIFIER), filter_name] + build_ridge_classifier(X_train, X_test,
                                                                                       y_train, y_test,
                                                                                       filter_spec_model_folder))

    #results.append([commons.fetch_model_name(constants.LINEAR_CLASSIFIER)] + build_linear_classifier(X_train, X_test,
                                                                                      # y_train, y_test,
                                                                                      # spec_model_folder))

    results.append(
        [commons.fetch_model_name(constants.LASSO_CLASSIFIER), filter_name] + build_lasso_classifier(X_train, X_test,
                                                                                       y_train, y_test,
                                                                                       filter_spec_model_folder))

    results.append(
        [commons.fetch_model_name(constants.ENET_CLASSIFIER), filter_name] + build_enet_classifier(X_train, X_test,
                                                                                        y_train, y_test,
                                                                                        filter_spec_model_folder))
    results.append(
        [commons.fetch_model_name(constants.LSVM_CLASSIFIER), filter_name] + build_lsvm_classifier(X_train, X_test,
                                                                                        y_train, y_test,
                                                                                        filter_spec_model_folder))
    results.append(
        [commons.fetch_model_name(constants.SVM_CLASSIFIER), filter_name] + build_svm_classifier(X_train, X_test,
                                                                                        y_train, y_test,
                                                                                        filter_spec_model_folder))
    results.append(
        [commons.fetch_model_name(constants.MLP_CLASSIFIER), filter_name] + build_mlp_classifier(X_train, X_test,
                                                                                    y_train, y_test,
                                                                                    filter_spec_model_folder))

    results.append(
        [commons.fetch_model_name(constants.GB_CLASSIFIER), filter_name] + build_gb_classifier(X_train, X_test,
                                                                                    y_train, y_test,
                                                                                    filter_spec_model_folder))
    return results, results_folder


def build_logistic_reg_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = LogisticRegressionCV(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.LOG_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)

def build_mlp_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = MLPClassifier(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.MLP_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)


def build_gb_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.GB_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)




def build_ridge_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = RidgeClassifier(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.RIDGE_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)



def build_enet_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = ElasticNet(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.ENET_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    return commons.show_scores(y_test, predictions)


def build_lasso_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = Lasso(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.LASSO_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    return commons.show_scores(y_test, predictions)


def build_lsvm_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.LSVM_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)

def build_svm_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = SVC(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.SVM_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)


def build_KNN_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.KNEIGH_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)

def build_random_forest_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.RAND_FOREST_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)

def build_decision_tree_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.DEC_TREE_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    # evaluate predictions
    return commons.show_scores(y_test, y_pred)



def build_xg_boost_classifier(X_train, X_test, y_train, y_test, model_folder):
    model = XGBClassifier(max_iter=10000)
    model.fit(X_train, y_train)
    # save the model
    commons.save_model(model, model_folder + constants.XGBOOST_CLASSIFIER)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    return commons.show_scores(y_test, predictions)  # Accuracy: 74.02%, F1 score: 63.33%, Mean MAE: 0.330 (0.036)
