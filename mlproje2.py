# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:17:12 2024

@author: hp
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')

# data frame dilimleme
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# numpy dizi array dönüşümü
X = x.values
Y = y.values

#Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() #obje oluşturduk
lin_reg.fit(X,Y) # x den y'yi öğren

#polynomial regression
# doğrusal olmayan (non linear) oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y) #x_poly yi y ye göre fit et 

# 4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y) 

# görselleştirme
plt.scatter(X,Y, color = 'red')
plt.plot(x,lin_reg.predict(X) , color = 'blue')
plt.show()

plt.scatter(X,Y) #görselleştir
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X))) 
plt.show()

plt.scatter(X,Y) #görselleştir
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X))) 
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg.predict(poly_reg.fit_transform([[6.6]])))            



