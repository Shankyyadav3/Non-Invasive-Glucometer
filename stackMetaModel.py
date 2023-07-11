import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, GridSearchCV, StratifiedKFold
import numpy as np
import commons, constants, feature_extractor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def perform_meta_modelling_classification(data_sheet, labels_sheet):
    data = commons.get_data_as_data_frame(data_sheet, None)
    labels = commons.get_data_as_data_frame(labels_sheet, None)
    # Assuming you have X (features) and y (labels) as your training data
    y = np.array(labels.values[:, 1].tolist())
    num_columns = data.shape[1]
    X = np.array(data.values[:, 1:num_columns])
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defining the base model
    gbc = GradientBoostingClassifier()

    # Defining the meta-model
    xgbc = XGBClassifier()

    # Defining the hyperparameter grid for GridSearchCV
    gbc_param_grid = {
        #'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1]
    }

    xgbc_param_grid = {
        #'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1],
        'max_depth': [3, 5, 7]
    }


    # Performing oversampling using ADASYN to balance the training dataset
    adasyn = ADASYN(random_state=42)
    X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)

    # Performing hyperparameter tuning using GridSearchCV for GBC Classifier
    gbc_grid_search = GridSearchCV(estimator=gbc, param_grid=gbc_param_grid, cv=5, scoring='accuracy')
    gbc_grid_search.fit(X_train_balanced, y_train_balanced)

    # Obtaining the best GBC model
    gbc_best_model = gbc_grid_search.best_estimator_

    # Performing hyperparameter tuning using GridSearchCV for XGBC Classifier
    xgbc_grid_search = GridSearchCV(estimator=xgbc, param_grid=xgbc_param_grid, cv=5, scoring='accuracy')
    #xgbc_grid_search.fit(X_train_balanced, y_train_balanced)


    # Storing the best models for the ADASYN SMOTE technique
    #best_models = {
    #    'base_model': gbc_best_model,
    #    'meta_model': xgbc_best_model
    #}
    best_models = {}

    # Evaluating the best models using k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []

    best_accuracy = 0.0
    best_f1_score = 0.0
    best_roc_auc = 0.0
    best_base_model = None
    best_meta_model = None

    for train_index, val_index in kf.split(X_train_balanced):
        X_train_fold, X_val_fold = X_train_balanced[train_index], X_train_balanced[val_index]
        y_train_fold, y_val_fold = y_train_balanced[train_index], y_train_balanced[val_index]

        # Training the base model on the training fold
        gbc_best_model.fit(X_train_fold, y_train_fold)

        # Predicting the probabilities of the positive class for each sample in the validation fold using the base model
        y_val_pred_base = gbc_best_model.predict_proba(X_val_fold)[:, 1]

        # Training the meta-model on the predictions of the base model
        xgbc_grid_search.fit(y_val_pred_base.reshape(-1, 1), y_val_fold)
        # Obtaining the best XGBC Classifier model
        xgbc_best_model = xgbc_grid_search.best_estimator_

        # Predicting the probabilities of the positive class for each sample in the validation fold using the meta-model
        y_val_pred_meta = xgbc_best_model.predict_proba(y_val_pred_base.reshape(-1, 1))[:, 1]

        # Computing evaluation metrics
        accuracy = accuracy_score(y_val_fold, y_val_pred_meta.round())
        f1 = f1_score(y_val_fold, y_val_pred_meta.round())
        roc_auc = roc_auc_score(y_val_fold, y_val_pred_meta)

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)

        # Updating the best model if the current model performs better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1_score = f1
            best_roc_auc = roc_auc
            best_base_model = gbc_best_model
            best_meta_model = xgbc_best_model

    # Storing the best performing model
    best_models['best_accuracy'] = best_accuracy
    best_models['best_f1_score'] = best_f1_score
    best_models['best_roc_auc'] = best_roc_auc
    best_models['base_model'] = best_base_model
    best_models['meta_model'] = best_meta_model

    # Saving the best models using pickle
    with open('best_base_model_gbc.pkl', 'wb') as file:
        pickle.dump(best_base_model, file)
    with open('best_meta_model_xgbc.pkl', 'wb') as file:
        pickle.dump(best_meta_model, file)



    # Evaluating the best models on the test set
    y_test_pred_base = best_base_model.predict_proba(X_test)[:, 1]
    y_test_pred_meta = best_meta_model.predict_proba(y_test_pred_base.reshape(-1, 1))[:, 1]

    # Computing evaluation metrics for the test set
    accuracy_test = accuracy_score(y_test, y_test_pred_meta.round())
    f1_test = f1_score(y_test, y_test_pred_meta.round())
    roc_auc_test = roc_auc_score(y_test, y_test_pred_meta)
    confusion_matrix_test = confusion_matrix(y_test, y_test_pred_meta.round())

    print("Test Set Results:")
    print("Accuracy:", accuracy_test)
    print("F1 Score:", f1_test)
    print("ROC AUC:", roc_auc_test)

    # Saving the evaluation results to a CSV file
    df_results = pd.DataFrame({
        'SMOTE Technique': ['ADASYN'],
        'Base Model': ['GBoost'],
        'Meta Model': ['XGBoost'],
        'Mean Accuracy': np.mean(accuracy_scores),
        'Mean F1 Score': np.mean(f1_scores),
        'Mean ROC AUC': np.mean(roc_auc_scores),
        'Test Accuracy': accuracy_test,
        'Test F1 Score': f1_test,
        'Test ROC Area': roc_auc_test
    })

    df_results.to_csv('gbc_xgbc_meta_evaluation_results.csv', index=False)

    # Printing the evaluation results
    print(df_results)


    print("Confusion Matrix:")
    print(confusion_matrix_test)
    return


def perform_meta_modelling_regression(data_sheet, labels_sheet):
    data = commons.get_data_as_data_frame(data_sheet, None)
    labels = pd.read_csv(labels_sheet, header=None)[0]
    # Assuming you have X (features) and y (labels) as your training data
    y = np.array(labels)
    num_columns = data.shape[1]
    X = np.array(data.values[:, 1:num_columns])


    # Define the base model (Gradient Boosting Regressor)
    base_model = GradientBoostingRegressor(random_state=42)

    # Define the meta-model (XGBoost Regressor)
    meta_model = xgb.XGBRegressor(random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid_base = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01]
    }

    param_grid_meta = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01]
    }

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create a dictionary to store the evaluation metrics for each fold
    evaluation_metrics = {'mse': []}

    best_mse = 100000.0
    best_models = {}

    for train_index, val_index in kfold.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Apply standard scaling to the training and validation sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Apply the base model (Gradient Boosting Regressor) with hyperparameter tuning
        gbr_grid_search = GridSearchCV(base_model, param_grid_base, scoring='neg_mean_squared_error', cv=3)
        gbr_grid_search.fit(X_train_scaled, y_train)
        best_base_model = gbr_grid_search.best_estimator_

        # Apply the meta-model (XGBoost Regressor) with hyperparameter tuning
        xgbr_grid_search = GridSearchCV(meta_model, param_grid_meta, scoring='neg_mean_squared_error', cv=3)
        xgbr_grid_search.fit(best_base_model.predict(X_train_scaled).reshape(-1, 1), y_train)
        best_meta_model = xgbr_grid_search.best_estimator_

        # Evaluate the best performing model on the validation set
        y_pred = best_meta_model.predict(best_base_model.predict(X_val_scaled).reshape(-1, 1))
        mse = mean_squared_error(y_val, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_models['base_model'] = best_base_model
            best_models['meta_model'] = best_meta_model
        evaluation_metrics['mse'].append(mse)

    # Print the mean evaluation metrics
    mean_mse = np.mean(evaluation_metrics['mse'])
    print(f"Mean MSE: {mean_mse:.4f}")

    # Save the best performing model using pickle
    filename = "best_model.pkl"
    with open('best_regress_base_model.pkl', 'wb') as file:
        pickle.dump(best_models['base_model'], file)

    with open('best_regress_meta_model.pkl', 'wb') as file:
        pickle.dump(best_models['meta_model'], file)

    # Prepare and save the results in a CSV file
    results = pd.DataFrame(evaluation_metrics)
    results.to_csv('regress_results.csv', index=False)
    return


def generate_test_features(test_data_sheet):
    test_df = commons.get_data_as_data_frame(test_data_sheet, None)
    feature_vector = []
    feature_set = []
    columns_to_extract = constants.TEST_DATA_COLUMN_NAMES
    df = test_df[columns_to_extract]
    patient_ids = list(set(df['Sample_No']))
    feature_set = []
    for patient_id in patient_ids:
        condition = df["Sample_No"] == patient_id  # define the condition
        patient_df = df.loc[condition, columns_to_extract]
        #fetch singular features
        duration = commons.compute_duration_from_timestamps(patient_df["Timestamps"])
        singular_features = [duration]
        for column_name in constants.TEST_SINGULAR_DATA_COLUMN_NAMES:
            singular_features.append(list(set(patient_df[column_name]))[0])
        feature_vector = []
        for col in constants.SENSOR_COLS:
            patient_values = patient_df.loc[:, col]
            patient_sensor_data = np.array(patient_values.values.tolist())

            stat_features = feature_extractor.get_stat_features(patient_sensor_data)
            # spatial features
            sensor_features = feature_extractor.obtain_spatial_freq_features_sensor_data(stat_features,
                                                                        'test_' + str(patient_id))
            feature_vector = feature_vector + sensor_features
        feature_set.append(singular_features + feature_vector)
    #print(feature_set)
    return feature_set


def perform_testing(test_data_sheet, models_dir):
    # Load the best base model
    print('doing testing')
    model_paths = commons.fetch_model_paths(models_dir)
    for path in model_paths:
        if path.find('base') >= 0 and path.find('.pkl') >= 0:
            base_model_path = path
        elif path.find('meta') >= 0 and path.find('.pkl') >= 0:
            meta_model_path = path
        else:
            continue
    with open(base_model_path, 'rb') as file:
        base_model = pickle.load(file)
    # Load the meta-model
    with open(meta_model_path, 'rb') as file:
        meta_model = pickle.load(file)

    test_features = generate_test_features(test_data_sheet)

    # Predict the probabilities of the positive class for the test record using the base model
    test_pred_base = base_model.predict_proba(test_features)[:, 1]

    # Predict the probabilities of the positive class for the test record using the meta-model
    test_pred_meta = meta_model.predict_proba(test_pred_base.reshape(-1, 1))[:, 1]

    # Round the predicted probability to the nearest integer (0 or 1)
    test_label = np.round(test_pred_meta).astype(int)
    final_message = ''
    if test_label[0] == 1:
        final_message = "you are likely to have Diabetes with 95.7 % accuracy"
    else:
        final_message = "you are likely to be non-Diabetic with 95.7 % accuracy"
    return final_message
