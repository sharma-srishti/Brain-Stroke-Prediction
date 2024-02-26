# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:29:16 2024

@author: hp
"""

import numpy as np
import pickle
import streamlit as st

# loading trained model 
loaded_model=pickle.load(open('C:/Users/hp/Brain Stroke Prediction/trained_model.sav','rb'))

# creating func for prediction
def brain_stroke_prediction(input_data):
    
    input_data_array = np.asarray(input_data)
    reshape_data = input_data_array.reshape(1,-1)
    prediction=loaded_model.predict(reshape_data)
    print(prediction)
    if(prediction[0]==1):
        return "The person has brain stroke"
    else:
        return  "The person does not has brain stroke"


def main():
    # giving a title
    st.title("Brain Stroke Preditcion Web App")
    
    # getting input data from user
    
    
    gender=st.text_input("Gender of the person,1 for Male and 0 for Female")
    age=st.text_input("Age of the person")
    hypertension=st.text_input("Is suffering from hypertension ,1 if yes else 0")
    heart_disease=st.text_input("Is suffering from heart disease ,1 if yes else 0")
    ever_married=st.text_input("Marital Status,1 for Married and 0 for Unmarried")
    work_type=st.text_input("Work type, 0 for Govt Job/1 for Private/2 for Self Employed/3 for children ")
    Residence_type=st.text_input("Residence type, 1 for Urban and 0 for Rural")
    avg_glucose_level=st.text_input("Average Glucose level")
    bmi=st.text_input("BMI value")
    smoking_status=st.text_input("Smoking Status,0 for Unknown/ 1 for formerly smoked/2 for never smoked/ 3 for smokes")
    
    # model prediction
    diagnosis=''
    
    # creating a button for prediction
    if st.button('Brain Stroke Test Result'):
        diagnosis=brain_stroke_prediction([gender,age,hypertension,
                                           heart_disease,ever_married,work_type,
                                           Residence_type,avg_glucose_level,bmi,
                                           smoking_status])
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    

    
    