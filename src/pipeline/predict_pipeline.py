import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        age: int,
        workclass: str,
        # fnlwgt: int,
        education: str,
        # education_num: int,
        marital_status: str,
        occupation: str,
        relationship: str,
        race: str,
        sex: str,
        capital_gain: int,
        capital_loss: int,
        hours_per_week: int,
        native_country: str):

        self.age = age
        self.workclass = workclass
        # self.fnlwgt = fnlwgt
        self.education = education
        # self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "workclass": [self.workclass],
                # "fnlwgt": [self.fnlwgt],
                "education": [self.education],
                # "education.num": [self.education_num],
                "marital.status": [self.marital_status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "sex": [self.sex],
                "capital.gain": [self.capital_gain],
                "capital.loss": [self.capital_loss],
                "hours.per.week": [self.hours_per_week],
                "native.country": [self.native_country]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


