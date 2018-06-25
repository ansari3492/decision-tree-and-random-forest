# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:06:18 2018

@author: Lenovo
"""

import pandas as pd

data=pd.read_csv("PastHires.csv")


features=data.iloc[:,0:6].values
labels=data["Hired"].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in [1,3,4,5]:
    features[:,i]=le.fit_transform(features[:,i])
    
features_df=pd.DataFrame(features)
labels=le.transform(labels)

    

#decision tree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(features,labels)
labels_predict=regressor.predict(features)

new_predict=regressor.predict([10,1,4,0,0,0])

new_predict1=regressor.predict([10,0,4,1,0,1])

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(features,labels)


new_predict2=rfr.predict([10,1,4,0,0,0])

new_predict3=rfr.predict([10,0,4,1,0,1])

