import pandas as pd
import numpy as np
import datetime
import time
#from dateutil.parser import parse
from impute_fun import impute
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from skmice import MiceImputer
#from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

def numFeat(data):
    data['Date'] = pd.to_datetime(data.Date)
    data['Date'] = data['Date'].view('int64') // pd.Timedelta(1, unit='s')
    # time.mktime(data['Date'].timetuple())
    # data['UnixTimel'] = (data['Date'] - datetime.datetime(1970, 1, 1, 0, 0, 0))
    # data['Date'] = data['UnixTimel']
    # data['Time'] = data['Time'].astype('datetime64[ns]')  
    data['Time'] = pd.to_datetime(data.Time).view('int64') // pd.Timedelta(1, unit='s') 
    
    return data[['cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'Time', 'visibility', 'windBearing', 'windSpeed', 'month','Date', 'Hour']]

def catFeat(data):
    return pd.get_dummies(data=data[['icon', 'precipType']])

dataset = pd.read_csv('../data/train_dataset.csv')



X = dataset[['icon', 'precipType', 'cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'Time', 'visibility', 'windBearing', 'windSpeed', 'month', 'Date', 'Hour']]
y = dataset[['Irradiance']]

X['precipType'] = X['precipType'].fillna('NA')

#imputer.fit(X[['cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'uvIndex', 'visibility', 'windBearing', 'windSpeed', 'month', 'Hour']])
#X[['cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'uvIndex', 'visibility', 'windBearing', 'windSpeed', 'month', 'Hour']] = imputer.fit_transform(X[['cloudCover', 'dewPoint', 'humidity', 'pressure', 'temperature', 'uvIndex', 'visibility', 'windBearing', 'windSpeed', 'month', 'Hour']])

features = FeatureUnion([('f1',FunctionTransformer(numFeat, validate=False)), ('f2',FunctionTransformer(catFeat, validate=False))])
pipeline = Pipeline( [('f', features)] )
pipeline.fit(X, y)
X = pipeline.transform(X)


#from skmice import MiceImputer
#imputer = MiceImputer()
#X, specs = imputer.transform(X, LinearRegression, 10)


#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
#imputer.fit(X[:, 0:14])
#X[:, 0:14] = imputer.transform(X[:, 0:14])


from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
X[:, 0:14] = KNN(k=3).complete(X[:,0:14])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


parameters = [];
parameters.append(('clf',  DecisionTreeRegressor(random_state = 0)))

    
model_pipeline = Pipeline(parameters)
model_pipeline.fit(X_train, y_train)

temp = np.array(X[13, :]).reshape((1, -1))

irradiance_pred = model_pipeline.predict(X_test)
score = mean_absolute_error(y_test, irradiance_pred)