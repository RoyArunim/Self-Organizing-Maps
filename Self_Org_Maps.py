# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:33:56 2023

@author: Dell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("D:/Masters/Self Organizing maps dataset/Self_Organizing_Maps/Credit_Card_Applications.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#splitting the X and Y not because it is supervised learning, but to store values if customer applications were approved or not
#no dependent variable



#Feature scaling

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
X=sc.fit_transform(X)


#Training the SOM
from minisom import MiniSom
som=MiniSom(10,10,15,sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train(X,100)

#Visualize the results
from pylab import bone, pcolor, colorbar, plot, show
bone() # just toinitialize the window
pcolor(som.distance_map().T) # mean inter neuron distance, transpose of that
colorbar()# for the legend of colors. the results show that the white boxes correspond to the highest inter neuron distance which means outliers and therefore frauds
'''
from here we can proceed, take the inverse transform of the winning nodes and see which nodes correspond to the frauds.
'''

'''
red circles customers who didnt get approval, green squares are customers who got approval
'''
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,  #the left coord of the squares
         w[1] + 0.5,
         markers[y[i]], #'o' or 's'
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#finding the frauds

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6,8)], mappings[(7,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))


#going from unsupervised learning to Supervised learning
#Verifying data using ANN

#Creating a matrix of features
customers = dataset.iloc[:,1:].values # except for the customer ids everything else is going in as features
isfraud=np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        isfraud[i]=1
#isfraud is the dpeendent variable
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
customers = sc.fit_transform(customers)



import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
ann=Sequential()

ann.add(Dense(units = 2, kernel_initializer='uniform', activation = 'relu', input_dim =15)) #2 units cause this is extension of the SOM, so 2 neuron then 1
ann.add(Dense(units = 1, kernel_initializer= 'uniform',activation='sigmoid'))
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(customers, isfraud, epochs = 2, batch_size=1) # dataset very very small. 

y_pred = ann.predict(customers)

y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred),axis =1)
    

#sort the numpy array
y_pred = y_pred[y_pred[:,1].argsort()]



