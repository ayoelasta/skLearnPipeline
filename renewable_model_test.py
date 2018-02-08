# This function provides a score using the mean absolute error for the model
# Input: raw dataset
# Output: model score

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_absolute_error

def numFeat(data):
    return data[['cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'visibility', 'windBearing', 'windSpeed', 'month', 'Hour']]

def catFeat(data):
    return pd.get_dummies(data=data[['icon', 'precipType']])


def get_model_score(dataset):
    X = dataset[['icon', 'precipType', 'cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'visibility', 'windBearing', 'windSpeed', 'month', 'Hour']]
    y = dataset[['Irradiance']]
    
    X['precipType'] = X['precipType'].fillna('NA')
    
    
    features = FeatureUnion([('f1',FunctionTransformer(numFeat, validate=False)), ('f2',FunctionTransformer(catFeat, validate=False))])
    pipeline = Pipeline( [('f', features)] )
    pipeline.fit(X, y)
    X = pipeline.transform(X)
    
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
    imputer.fit(X[:, 0:10])
    X[:, 0:10] = imputer.transform(X[:, 0:10])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
     
    parameters = [];
    parameters.append(('clf',  DecisionTreeRegressor(random_state = 0)))
        
    model_pipeline = Pipeline(parameters)
    model_pipeline.fit(X_train, y_train)
    irradiance_pred = model_pipeline.predict(X_test)
    #score = model_pipeline.score(X_test, y_test)
    score = mean_absolute_error(y_test, irradiance_pred, multioutput='raw_values')
   
    return score


