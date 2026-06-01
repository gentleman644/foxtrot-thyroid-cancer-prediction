from joblib import load

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

class model():
    #Purpose: The first iteration of the Thyroid Cancer Prediction Model.
    #         It takes in parameters and make a prediction based on the parameters
    #Author(s): Tim Liu
    def __init__(self):
        self.current_model = load("joblib/model.joblib")
        self.standard = load("joblib/standardization.joblib")

        self.Age = 76
        self.Smoking = True
        self.Obesity = True
        self.Family_History = True
        self.Gender = 'Male'
        self.Thyroid_Cancer_Risk = 'High'
        self.T4_Level = 5.0
        self.T3_Level = 1.0
        self.TSH_Level = 3.0
        self.Nodule_Size = 2.0
        self.Country = "Russia"
        self.Ethnicity = "Asian"
        self.Radiation_Exposure = True
        self.Iodine_Deficiency = True
        self.Diabetes = True

    def predict(self):
        #Purpose: predicts based on the current parameters
        #Author(s): Tim Liu
        predict_input = pd.DataFrame([{
            "Age" : int(self.Age),
            "Smoking" : int(self.Smoking),
            "Obesity" : int(self.Obesity),
            "TSH_Level_per_T3_Level": self.TSH_Level / self.T3_Level,
            "T4_Level_per_T3_Level": self.T4_Level / self.T3_Level,
            "Nodule_Size_per_TSH_Level" : self.Nodule_Size / self.TSH_Level,
            "Nodule_Size_per_T3_Level" : self.Nodule_Size / self.T3_Level,
            "Nodule_Size_per_T4_Level" : self.Nodule_Size / self.T4_Level,
            "T3_Level_by_TSH_Level" : self.T3_Level / self.TSH_Level,
            "T4_Level_by_TSH_Level" : self.T4_Level / self.TSH_Level,
            "T4_Level_by_T3_Level" : self.T4_Level / self.T3_Level,
            "Nodule_Size_by_TSH_Level" : self.Nodule_Size / self.TSH_Level,
            "Nodule_Size_by_T3_Level" : self.Nodule_Size / self.T3_Level,
            "Nodule_Size_by_T4_Level" : self.Nodule_Size / self.T4_Level,
            "Gender_1.0" : 0,
            "Country_1.0" : 0,
            "Country_2.0" : 0,
            "Country_3.0" : 0,
            "Country_4.0" : 0,
            "Country_5.0" : 0,
            "Country_6.0" : 0,
            "Country_7.0" : 0,
            "Country_8.0" : 0,
            "Country_9.0" : 0,
            "Ethnicity_1.0" : 0,
            "Ethnicity_2.0" : 0,
            "Ethnicity_3.0" : 0,
            "Ethnicity_4.0" : 0,
            "Family_History_1.0" : int(self.Family_History),
            "Radiation_Exposure_1.0" : int(self.Radiation_Exposure),
            "Iodine_Deficiency_1.0" : int(self.Iodine_Deficiency),
            "Diabetes_1.0" : int(self.Diabetes),
            "Thyroid_Cancer_Risk_1.0" : 0,
            "Thyroid_Cancer_Risk_2.0" : 0
        }])

        match self.Gender:
            case 'Male'| 'male':
                predict_input['Gender_1.0'] = True
            case 'Female' | 'female':
                predict_input['Gender_1.0'] = False

        match self.Country:
            case 'Germany':
                predict_input['Country_1.0'] = 1
            case 'Nigeria':
                predict_input["Country_2.0"] = 1
            case 'India':
                predict_input["Country_3.0"] = 1
            case 'UK':
                predict_input["Country_4.0"] = 1
            case 'South Korea':
                predict_input["Country_5.0"] = 1
            case 'Brazil':
                predict_input["Country_6.0"] = 1
            case 'China':
                predict_input["Country_7.0"] = 1
            case 'US':
                predict_input["Country_8.0"] = 1
            case 'Japan':
                predict_input["Country_9.0"] = 1

        match self.Ethnicity:
            case 'Hispanic':
                predict_input['Ethnicity_1.0'] = 1
            case 'Asian':
                predict_input['Ethnicity_2.0'] = 1
            case 'African':
                predict_input['Ethnicity_3.0'] = 1
            case 'Middle Eastern':
                predict_input['Ethnicity_4.0'] = 1

        match self.Thyroid_Cancer_Risk:
            case 'Low':
                predict_input['Thyroid_Cancer_Risk_1.0'] = 1
            case 'Medium':
                predict_input['Thyroid_Cancer_Risk_2.0'] = 1

        standized_predict_input = self.standard.transform(predict_input)
        predict_output = self.current_model.predict_proba(standized_predict_input)
        return("Malignant" if (predict_output[:,1] > 0.4) else "Benign")