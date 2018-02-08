# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy import stats

df = pd.read_csv("C:\\Users\\AyodejiAkiwowo\\Documents\\Ayo\\RenewablesPython\\data\\train_dataset.csv")

df.head()

