
import os
import sys
from dataclasses import dataclass

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "KNeighbors Classifier": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBClassifier": xgb.XGBClassifier(),
                "LightGBM": lgb.LGBMClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            params = {
                "KNeighbors Classifier": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'C': [0.1, 1],
                    'solver': ['lbfgs']
                },
                "Naive Bayes": {
                    'var_smoothing': [1e-9, 1e-8]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy']
                },
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'criterion': ['gini', 'entropy']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                },
                "LightGBM": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8],
                    'learning_rate': [0.1, 0.01],
                    'iterations': [50, 100]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'subsample': [0.8, 0.9],
                    'n_estimators': [50, 100]
                }
            }
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            return best_model, acc
            



            
        except Exception as e:
            raise CustomException(e,sys)