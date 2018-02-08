# This script tests the model pipeline in renewable_model_auto function. 

from renewable_model_auto import create_model
from renewable_model_test import get_model_score
import pandas as pd


dataset = pd.read_csv('../data/train_dataset.csv')
model = create_model(dataset)
#
#temp = [0.460248,	57.58,	0.85,	1012.94,	62.22,	0.685525,	8.53,	216,	11.83,	7,	20,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]
#temp = np.array(temp).reshape((1, -1))
#y_pred = model.predict(temp)
#print(y_pred)

#print (get_encoded_feature([0.27,0.82,49.49,9.49,12,11, 'partly-cloudy-day', 'rain']))

p = get_model_score(dataset)
print(p)