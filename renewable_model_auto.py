import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.externals import joblib

def numFeat(data):
    data['Date'] = pd.to_datetime(data.Date)
    data['Date'] = data['Date'].view('int64') // pd.Timedelta(1, unit='s')
    data['Time'] = pd.to_datetime(data.Time).view('int64') // pd.Timedelta(1, unit='s') 
    
    return data[['cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'Time', 'visibility', 'windBearing', 'windSpeed', 'month','Date', 'Hour']]

def catFeat(data):
    return pd.get_dummies(data=data[['icon', 'precipType']])


def create_model(dataset):
    X = dataset[['icon', 'precipType', 'cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'Time', 'visibility', 'windBearing', 'windSpeed', 'month', 'Date', 'Hour']]
    y = dataset[['Irradiance']]
    
    X['precipType'] = X['precipType'].fillna('NA')
    
    
    features = FeatureUnion([('f1',FunctionTransformer(numFeat, validate=False)), ('f2',FunctionTransformer(catFeat, validate=False))])
    pipeline = Pipeline( [('f', features)] )
    pipeline.fit(X, y)
    X = pipeline.transform(X)
    
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy='median', axis=0, copy=False)
    imputer.fit(X[:, 0:14])
    X[:, 0:14] = imputer.transform(X[:, 0:14])
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
     
    parameters = [];
    parameters.append(('clf',  DecisionTreeRegressor(random_state = 0)))
        
    model_pipeline = Pipeline(parameters)
    model_pipeline.fit(X_train, y_train)
    
    joblib.dump(model_pipeline, 'model.pkl') 
    
    return model_pipeline