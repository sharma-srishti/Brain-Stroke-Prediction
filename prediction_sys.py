
import numpy as np
import pickle 

# loading trained model 
loaded_model=pickle.load(open('C:/Users/hp/Brain Stroke Prediction/trained_model.sav','rb'))

input_data =   (0,56,0,0,1,1,0,246,34,2)
input_data_array = np.asarray(input_data)
reshape_data = input_data_array.reshape(1,-1)
prediction=loaded_model.predict(reshape_data)
print(prediction)
if(prediction[0]==1):
    print("The person has brain stroke")
else:
    print("The person does not has brain stroke")