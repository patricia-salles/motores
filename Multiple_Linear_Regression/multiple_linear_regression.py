# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 00:18:33 2019

@author: pati_
"""


'''Multiple Linear Regression'''
'''
Variáveis categóricas precisam ser transformadas em dummies
As linhas viram colunas que são preenchidas com 1 (sim) 0 (não)
Mas não precisa incluir todas as variáveis dummies no modelo, você
sempre inclui o número de dummies -1, ou seja, se criou 2 dummies,
inclua uma, se é 100 dummies, inclua 99. Esse passo evita problemas
com multicolinearidade entre as variáveis

P-valor
O valor-p para cada termo testa a hipótese nula de que o coeficiente 
é igual a zero (sem efeito). Um valor-p baixo (< 0,05) indica que 
você pode rejeitar a hipótese nula. Em outras palavras, uma preditora 
que tenha um valor-p baixo provavelmente será uma adição significativa 
ao seu modelo, porque as alterações no valor da preditora estão 
relacionadas a alterações na variável resposta.

5 métodos para criar o modelo

All-in
Backward elimination
Forward selection
Bidirectional elimination
Score comparison

2,3,4 são stepwise
Backward: 
1 Seta um pvalor com nível de significancia
2 Rodar o modelo com todas as variáveis e ver qual dela tem maior pvalor
3 Removar essa variável
4 Rodar o modelo sem essa variável
Repetir os passos 2,3,4 até todas as variáveis com alto pvalor sumirem

Forward
1 Seta um pvalor com nível de significancia
2 Rodar o modelo e selecionar a variável com o menor pvalor
3 Rodar o modelo com essa variável e mais uma
4 Manter a variável com o menor p-valor
Repetir os passos 3,4

All Possible Models
1 Seleciona um critério de acurácia do modelo (ex Akaike criterio)
2 Construa todos os modelos possíveis 2^n-1 => 10 colunas = 1023 modelos
3 Seleciona o modelo com o melhor critério

'''


import pandas as pd
import numpy as np
cd C:\Users\pati_\Documents\Motor\Machine Learning A-Z\Multiple_Linear_Regression

df=pd.read_csv("50_Startups.csv", sep=',')

###selecionar as variáveis x e y
x=df.iloc[:, :-1].values
y=df.iloc[:, 4].values

###transformar as variáveis categóricas em dummies
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_x = LabelEncoder()
x[:, 3] = labelenconder_x.fit_transform(x[:,3])

###transforma a categórica em dummy
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

###excluir uma das variáveis dummies-evita multicolinearidade
x=x[:,1:]

###separar o dataframe em treino e teste (proporção de 80% e 20%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

###fit multiple linear regression
from sklearn.linear_model import LinearRegression
###criar o objeto da regressão linear
regressor=LinearRegression()
regressor.fit(x_train, y_train)