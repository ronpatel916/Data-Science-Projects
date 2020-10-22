import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from itertools import combinations

data = pd.read_csv('sphist.csv')

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date',ascending = True)


data['5 day rolling avg'] = data['Close'].shift(1).rolling(5).mean()
data['30 day rolling avg'] = data['Close'].shift(1).rolling(30).mean()
data['365 day rolling avg'] = data['Close'].shift(1).rolling(365).mean()
data['5_to_365 day price ratio'] = data['5 day rolling avg']/data['365 day rolling avg']
data['5 day rolling std'] = data['Close'].shift(1).rolling(5).std()
data['365 day rolling std'] = data['Close'].shift(1).rolling(365).std()
data['5_to_365 day std ratio'] = data['5 day rolling std']/data['365 day rolling std']
data['5 day avg volume'] = data['Volume'].shift(1).rolling(5).mean()
data['365 day avg volume'] = data['Volume'].shift(1).rolling(365).mean()
data['5_to_365 day avg volume'] = data['5 day avg volume']/data['365 day avg volume']

data['5 day volume std'] = data['Volume'].shift(1).rolling(5).std()
data['365 day volume std'] = data['Volume'].shift(1).rolling(365).std()
data['5_to_365 day std volume ratio'] = data['5 day volume std']/data['365 day volume std']

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Day of Week'] = data['Date'].dt.dayofweek

data = data[data['Date'] > datetime(year=1951,month=1,day=2)]
data = data.dropna(axis = 0)

train = data[data['Date'] < datetime(year = 2013,month = 1, day =1)]
test = data[data['Date'] >= datetime(year = 2013,month = 1, day =1)]


features = ['5 day rolling avg','30 day rolling avg','5_to_365 day price ratio']

lr = LinearRegression()
lr.fit(train[features],train['Close'])

predictions = lr.predict(test[features])

simple_model_mae = mean_absolute_error(test['Close'],predictions)


all_features = train.columns[7:]

mae_values = {}

for i in range(3,4):
    for combo in combinations(all_features,i):
        features = list(combo)
        feature_label = ''
        for feature in features:
            feature_label += feature + ", "
        feature_label = feature_label[:-2]
        
        lr = LinearRegression()
        lr.fit(train[features],train['Close'])
        prediction = lr.predict(test[features])
        mae = mean_absolute_error(test['Close'],prediction)
        mae_values[feature_label] = mae

    
min_mae = min(mae_values, key = mae_values.get)
print(min_mae, ": ", mae_values[min_mae])