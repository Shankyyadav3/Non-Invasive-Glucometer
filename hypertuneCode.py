import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
#from dbn.tensorflow import SupervisedDBNClassification
import commons

def perform_hyperparameter_tuning(data_sheet, labels_sheet):
    data = commons.get_data_as_data_frame(data_sheet, None)
    labels = commons.get_data_as_data_frame(labels_sheet, None)
    # Assuming you have X (features) and y (labels) as your training data
    y = np.array(labels.values[:, 1].tolist())
    num_columns = data.shape[1]
    X = np.array(data.values[:, 1:num_columns])

    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defining the models
    gbc = GradientBoostingClassifier()
    xgbc = XGBClassifier()
    dtc = DecisionTreeClassifier()
    dbn = Pipeline(steps=[('rbm', BernoulliRBM(random_state=42)), ('classifier', LogisticRegression())])

    # Defining the hyperparameter grid for GridSearchCV
    gbc_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 7]
    }

    xgbc_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 7]
    }

    dtc_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    #dbn_param_grid = {
    #    'n_hidden_neurons': [100, 200, 300],
    #    'learning_rate': [0.1, 0.01, 1],
    #    'n_epochs': [10, 20, 30]
    #}
    dbn_param_grid = {
        'rbm__n_components': [100, 200, 300],
        'rbm__learning_rate': [0.1, 0.01],
        'classifier__C': [0.1, 1, 10]
    }

    # Dictionary to store the best models for each SMOTE technique
    best_models = {}

    # List to store the evaluation metrics
    evaluation_results = []

    # List of SMOTE techniques to apply
    smote_techniques = [
        #("SMOTE", SMOTE(random_state=42)),
        #("Borderline-SMOTE", BorderlineSMOTE(random_state=42)),
        ("ADASYN", ADASYN(random_state=42)),
    ]

    print("smote")
    # Loop over each SMOTE technique
    for smote_name, smote in smote_techniques:
        # Performing oversampling using the current SMOTE technique to balance the training dataset
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Performing hyperparameter tuning using GridSearchCV for Gradient Boosting Classifier
        gbc_grid_search = GridSearchCV(estimator=gbc, param_grid=gbc_param_grid, cv=5, scoring='accuracy')
        gbc_grid_search.fit(X_train_balanced, y_train_balanced)

        # Obtaining the best Gradient Boosting Classifier model
        gbc_best_model = gbc_grid_search.best_estimator_

        print("hypertune")

        # Performing hyperparameter tuning using GridSearchCV for XGBoost Classifier
        xgbc_grid_search = GridSearchCV(estimator=xgbc, param_grid=xgbc_param_grid, cv=5, scoring='accuracy')
        xgbc_grid_search.fit(X_train_balanced, y_train_balanced)

        # Obtaining the best XGBoost Classifier model
        xgbc_best_model = xgbc_grid_search.best_estimator_

        # Performing hyperparameter tuning using GridSearchCV for Decision Tree Classifier
        dtc_grid_search = GridSearchCV(estimator=dtc, param_grid=dtc_param_grid, cv=5, scoring='accuracy')
        dtc_grid_search.fit(X_train_balanced, y_train_balanced)

        # Obtaining the best Decision Tree Classifier model
        dtc_best_model = dtc_grid_search.best_estimator_

        # Performing hyperparameter tuning using GridSearchCV for DBN Classifier
        dbn_grid_search = GridSearchCV(estimator=dbn, param_grid=dbn_param_grid, cv=5, scoring='accuracy')
        dbn_grid_search.fit(X_train_balanced, y_train_balanced)

        # Obtaining the best DBN Classifier model
        dbn_best_model = dbn_grid_search.best_estimator_

        # Storing the best models for the current SMOTE technique
        best_models[smote_name] = {
        'gbc': gbc_best_model,
        'xgbc': xgbc_best_model,
        'dtc': dtc_best_model,
        'dbn': dbn_best_model
        }
        print("k-fold cross val")
        # Evaluating the best models using k-fold cross-validation
        gbc_scores = cross_val_score(gbc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        gbc_f1_scores = cross_val_score(gbc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1_macro')
        gbc_roc_auc_scores = cross_val_score(gbc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')

        xgbc_scores = cross_val_score(xgbc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        xgbc_f1_scores = cross_val_score(xgbc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1_macro')
        xgbc_roc_auc_scores = cross_val_score(xgbc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')

        dtc_scores = cross_val_score(dtc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        dtc_f1_scores = cross_val_score(dtc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1_macro')
        dtc_roc_auc_scores = cross_val_score(dtc_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')

        dbn_scores = cross_val_score(dbn_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
        dbn_f1_scores = cross_val_score(dbn_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1_macro')
        dbn_roc_auc_scores = cross_val_score(dbn_best_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')

        evaluation_results.append({
            'SMOTE Technique': smote_name,
            'Classifier': 'Gradient Boosting',
            'Mean Accuracy': np.mean(gbc_scores),
            'Mean F1 Score': np.mean(gbc_f1_scores),
            'Mean ROC Area': np.mean(gbc_roc_auc_scores)
        })
        evaluation_results.append({
            'SMOTE Technique': smote_name,
            'Classifier': 'XGBoost',
            'Mean Accuracy': np.mean(xgbc_scores),
            'Mean F1 Score': np.mean(xgbc_f1_scores),
            'Mean ROC Area': np.mean(xgbc_roc_auc_scores)
        })
        evaluation_results.append({
            'SMOTE Technique': smote_name,
            'Classifier': 'Decision Tree',
            'Mean Accuracy': np.mean(dtc_scores),
            'Mean F1 Score': np.mean(dtc_f1_scores),
            'Mean ROC Area': np.mean(dtc_roc_auc_scores)
        })
        evaluation_results.append({
            'SMOTE Technique': smote_name,
            'Classifier': 'DBN',
            'Mean Accuracy': np.mean(dbn_scores),
            'Mean F1 Score': np.mean(dbn_f1_scores),
            'Mean ROC Area': np.mean(dbn_roc_auc_scores)
        })

    # Saving the best performing models using pickle
    for key in best_models.keys():
        with open(key+'_sp_hypertune_best_model.pkl', 'wb') as f:
            pickle.dump(best_models[key], f)

    # Saving the evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_csv('sp_hypertune_evaluation_results.csv', index=False)
    return
