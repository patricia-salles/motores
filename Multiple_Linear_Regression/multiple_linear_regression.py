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

### Predict the test set results
y_pred=regressor.predict(x_test)

###Backward elimination
'''
Linear models with independently and identically distributed errors, and for 
errors with heteroscedasticity or autocorrelation. This module allows 
estimation by ordinary least squares (OLS), weighted least squares (WLS), 
generalized least squares (GLS), and feasible generalized least squares 
with autocorrelated AR(p) errors.

OLS chooses the parameters of a linear function of a set of explanatory 
variables by the principle of least squares: minimizing the sum of the 
squares of the differences between the observed dependent variable (values 
of the variable being predicted) in the given dataset and those predicted by 
the linear function.

Geometrically, this is seen as the sum of the squared distances, parallel to 
the axis of the dependent variable, between each data point in the set and 
the corresponding point on the regression surface – the smaller the differences,
the better the model fits the data. 
'''
import statsmodels.api as sm
back=np.append(arr=np.ones((50,1)).astype(int), values = x, axis =1)
back_opt=back[:, [0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y, exog=back_opt).fit() 
regressor_ols.summary()

###remove index 2 because p value
back_opt=back[:, [0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y, exog=back_opt).fit() 
regressor_ols.summary()

###remove index 1 because p value
back_opt=back[:, [0,3,4,5]]
regressor_ols=sm.OLS(endog=y, exog=back_opt).fit() 
regressor_ols.summary()

###remove index 4 because p value
back_opt=back[:, [0,3,5]]
regressor_ols=sm.OLS(endog=y, exog=back_opt).fit() 
regressor_ols.summary()

###remove index 5 because p value
back_opt=back[:, [0,3]]
regressor_ols=sm.OLS(endog=y, exog=back_opt).fit() 
regressor_ols.summary()

###create a new dataframe with only important variables after backward elimination
X_new = back[:, [0,3]] 
y_pred_ols = regressor_ols.predict(X_new)

'''

if you are also interested in some automatic implementations of Backward 
Elimination in Python, please find two of them below:

Backward Elimination with p-values only:

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
Backward Elimination with p-values and Adjusted R Squared:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

