import sys
from dataclasses import dataclass

from imblearn.over_sampling import ADASYN
from collections import Counter


import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns =['age',
                                'fnlwgt',
                                'education.num',
                                'capital.gain',
                                'capital.loss',
                                'hours.per.week']
            
            categorical_columns = ['workclass',
                                'marital.status',
                                'occupation',
                                'relationship',
                                'race',
                                'sex',
                                'native.country',
                                ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            train_df['income'] = train_df['income'].map({'<=50K': 0, '>50K': 1})
            test_df['income'] = test_df['income'].map({'<=50K': 0, '>50K': 1})

            logging.info("Obtaining preprocessing object")

            train_df = train_df.replace("?","Unknown")
            test_df = test_df.replace("?","Unknown")
            train_df.drop("education", axis=1, inplace = True)
            test_df.drop("education", axis=1, inplace = True)

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="income"
            # numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # ...existing code...

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Oversampling the train dataset using ADASYN
            # ada = ADASYN(random_state=130)
            # X_train, y_train = ada.fit_resample(X_train, y_train)

            # Reshape target arrays to column vectors to avoid zero-dimensional errors
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Now ensure dtype is float64
            input_feature_train_arr = np.array(input_feature_train_arr, dtype=np.float64)
            input_feature_test_arr = np.array(input_feature_test_arr, dtype=np.float64)

            print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            print("target_feature_train_arr shape:", target_feature_train_arr.shape)
            print("input_feature_test_arr shape:", input_feature_test_arr.shape)
            print("target_feature_test_arr shape:", target_feature_test_arr.shape)
            
            # Convert target to 1D array for ADASYN
            target_feature_train_arr = np.array(target_feature_train_df).ravel()  
            target_feature_test_arr = np.array(target_feature_test_df).ravel()

            # 🔹 Apply ADASYN oversampling
            ada = ADASYN(random_state=130)
            input_feature_train_arr, target_feature_train_arr = ada.fit_resample(
                input_feature_train_arr, target_feature_train_arr
            )

            print("After ADASYN:")
            print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            print("target_feature_train_arr shape:", target_feature_train_arr.shape)

            # Combine for saving
            train_arr = np.concatenate(
                (input_feature_train_arr, target_feature_train_arr.reshape(-1, 1)), axis=1
            )
            test_arr = np.concatenate(
                (input_feature_test_arr, target_feature_test_arr.reshape(-1, 1)), axis=1
            )

            # train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            # logging.info(f"Train array shape: {train_arr.shape}")

            # test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)
            # logging.info(f"Test array shape: {test_arr.shape}")

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
