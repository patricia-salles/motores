# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:37:28 2019

@author: pati_
"""

 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cd C:\Users\pati_\Documents\Motor\motores\Artificial_Neural_Networks

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
##exclui as features de ID
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical  - transformando os valores categóricos, p.ex Feminino/Masculino 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - padronizar os dados
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Neural Network!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential  ### iniciar a rede neural
from keras.layers import Dense   ### adicionar as camadas ocultas

# Initialising the neural network
classifier = Sequential()

'''
Step1 Inicializar os pesos randomicamente para valores próximo de zero
Step2 colocar a primeira observação do dataset na camada de input, cada variável 
em um nó de input - nós temos 11 variáveis, assim teremos 11 input nodes
Step3 Forward propagation - da esquerda pra direita
Step4 Compara o valor predito com o valor atual e gera o erro
Step5 Backpropagation - da direita pra esquerda
Step 6 Repetir 1 ao 5 e atualiza os pesos depois de cada observação (Reinforcement Learning)
Repetit 1 ao 5 mas só atualiza os pesos depois "batch observations" (Batch Learning)

'''
## adicionar a camada de input e o primeira camada oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))   

 ## output_dim - quantos nós tem nas camadas ocultas - "dicas/experimentação"
##pode fazer uma média entre quantidade de variáveis e qtd de resposta no output
## init = uniform garante que os pesos iniciarão em números próximos de zero
##input_dim - número de variáveis independentes, número de nós no camada de input

## adicionar a segunda camada oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))   

## adicionar camada de output
classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))   
## units=1 one vs all método - se a resposta tiver mais de duas categorias de resposta, p.ex 4 units=4
## activation = sigmoid, mas se tiver mais de duas categorias de resposta, aí é soft max function

## compilar a rede neural
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## metrics = similar a função de perda, foi usada a acurácia
## loss = o objeto que o modelo quer minimizar - binary_crossentropy pq temos duas categorias de resposta
## optmizer = adam = um algoritmo de 1ª ordem baseada em gradiente de funções objetivas estocásticas
## usado para problemas com muito ruido e gradientes esparsos

## Fit modelo de rede neural na base de treinamento
classifier.fit()


# Fazendo as predições da base de teste
y_pred=classifier.predict(X_test)