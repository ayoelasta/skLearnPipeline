import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib

#value is weather condition
def get_encoded_feature(observation):
    icon_encode = {
                'clear-day' : [1, 0, 0, 0, 0, 0, 0, 0, 0],
                'clear-night' : [0, 1, 0, 0, 0, 0, 0, 0, 0],
                'cloudy' : [0, 0, 1, 0, 0, 0, 0, 0, 0],
                'fog' : [0, 0, 0, 1, 0, 0, 0, 0, 0],
                'partly-cloudy-day' : [0, 0, 0, 0, 1, 0, 0, 0, 0],
                'partly-cloudy-night' : [0, 0, 0, 0, 0, 1, 0, 0, 0],
                'rain' : [0, 0, 0, 0, 0, 0, 1, 0, 0],
                'snow' : [0, 0, 0, 0, 0, 0, 0, 1, 0],
                'wind' : [0, 0, 0, 0, 0, 0, 0, 0, 1]
            }
    
    precipType_encode = {
            'rain' : [1, 0 , 0],
            'snow' : [0, 1, 0],
            }
     
    encoded_observation =  []
    encoded_observation.append(observation[0:6])
    encoded_observation.append(icon_encode[observation[6]])
    encoded_observation.append(precipType_encode[observation[7]])
    
    flat_list = [item for sublist in encoded_observation for item in sublist]
    
    return flat_list

def numFeat(data):
    return data[['cloudCover', 'humidity', 'temperature', 'visibility', 'month', 'Hour']]

def get_encoded_data(data):
     
    for index, row in data.iterrows():
        print(row)

def create_model(dataset):
    
    X = dataset[['icon', 'precipType', 'cloudCover', 'humidity', 'temperature', 'visibility', 'month', 'Hour']]
    y = dataset[['Irradiance']]

    X['precipType'] = X['precipType'].fillna('rain')

    features = FeatureUnion([('f1',FunctionTransformer(numFeat, validate=False)),
                         ('f2',FunctionTransformer(catFeat, validate=False))] )
    encoder_pipeline = Pipeline( [('f', features)] )
    
    encoder_pipeline.fit(X, y)
    X_new = encoder_pipeline.transform(X)
    
    parameters = [];
    parameters.append(('imp', Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)))
    #parameters.append(('clf', BaggingRegressor(n_estimators=50, random_state = 0)))
    parameters.append(('clf',  DecisionTreeRegressor(random_state = 0)))
        
    model_pipeline = Pipeline(parameters)
    model_pipeline.fit(X_new, y)
        
    joblib.dump(model_pipeline, 'model.pkl') 
    
    return model_pipeline
    
