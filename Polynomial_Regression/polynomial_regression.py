# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:12:11 2019

@author: pati_
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cd C:\Users\pati_\Documents\Motor\Machine Learning A-Z\Multiple_Linear_Regression

df=pd.read_csv("Position_Salaries.csv", sep=',')

###selecionar as variáveis x e y - coluna Position corresponde a coluna Level
x=df.iloc[:, 1:2].values
y=df.iloc[:, 2].values

'''
não precisa quebrar a base em treino e teste, porque tem poucos valores e 
porque você deseja prever o modelo com muita acurácia e precisa de todos valores

não precisa fazer feature scaling porque vamos criar polinomios a partir
da função regressão linear múltipla
'''

###Fit linear regression to dataframe - apenas para comparação
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x, y)

###Fit Polynomial Regression to dataframe
###criação da matriz "polinomial", associados com termos polinomiais
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)  ### default é 2
x_poly=poly_reg.fit_transform(x)

###x_poly criou 3 'variáveis', uma em que constante é x1=1, x1 e x1^2

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly, y)

###Visualizar resultados regressão linear 
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

#### linear regression não foi um bom modelo para essa base de dados

###Visualizar resultados regressão polinomial
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg2.predict(x_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

'''
se aumentar o grau do polinomio o modelo vai ter um aumento na sua acurácia, 
nesse caso
'''
