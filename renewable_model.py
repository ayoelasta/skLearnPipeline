import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib

def numFeat(data):
    return data[['cloudCover', 'humidity', 'temperature', 'visibility', 'month', 'Hour']]

def catFeat(data):
    return pd.get_dummies(data[['icon', 'precipType']])

def create_model(dataset):
    
    X = dataset[['icon', 'precipType', 'cloudCover', 'humidity', 'temperature', 'visibility', 'month', 'Hour']]
    y = dataset[['Irradiance']]

    X['precipType'] = X['precipType'].fillna('rain')

    features = FeatureUnion([('f1',FunctionTransformer(numFeat, validate=False)), ('f2',FunctionTransformer(catFeat, validate=False))])
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
    
