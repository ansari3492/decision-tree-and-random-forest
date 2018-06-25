# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:12:06 2018

@author: Lenovo
"""
import pandas as pd
dataset=pd.read_csv("Auto_mpg.txt",delim_whitespace=True)


dataset.columns=["mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name" ]


#removing ? values and fill with the mode
dataset["horsepower"][dataset["horsepower"] == "?"]=dataset["horsepower"].mode()[0]
dataset["horsepower"] =dataset["horsepower"].astype('float')

features=dataset.iloc[:,1:-1].values
labels=dataset["mpg"].values
data=pd.DataFrame(features)


maxi=max(dataset["mpg"])

print(dataset["car name"][dataset["mpg"] == maxi])

#splitting the data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)



#decision tree
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(features_train,labels_train)
labels_predict=regressor.predict(features_test)
score=regressor.score(features_test,labels_test)

#random forest tree amking
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=10,random_state=0)
rfr.fit(features_train,labels_train)
labels_predict2=rfr.predict(features_test)
score2=rfr.score(features_test,labels_test)

#prediction
prediction1=regressor.predict([6,215,215,2630,22.2,80,3])
prediction2=rfr.predict([6,215,215,2630,22.2,80,3])




















#visual understanding
features_grid=np.arange(min(features),max(features),0.1)
features_grid=features_grid.reshape(-1,1)
plt.scatter(features,labels,color='CMY')
plt.plot(features_grid,regressor.predict(features_grid),color = 'blue')
plt.title("age vs length(decision tree regression)")
plt.xlabel("age")
plt.ylabel("length")
plt.show()