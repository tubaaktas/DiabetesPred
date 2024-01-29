#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,
# NEW_AGE_CAT_old, 1 0,NEW_AGE_CAT_young, 0 1, 7
# ,NEW_BMI_Healthy,NEW_BMI_Overweight,NEW_BMI_Obese, 5
# NEW_GLUCOSE_Prediabetes,NEW_GLUCOSE_Diabetes, 1
# NEW_BLOODPRESSURE_hs1,NEW_BLOODPRESSURE_hs2, 2
# NEW_INSULIN_Normal 4

#3,128,78,0,0.0,21.1,0.268,55,0,0,1,0,0,0,0,0,0,0



import joblib
import pandas as pd
import numpy as np

data = pd.read_csv('ara_yuz/pipeline/X_train.csv')

model = joblib.load('ara_yuz/pipeline/model.joblib')
model.predict_proba(np.array(data.iloc[85]).reshape(1, -1))
