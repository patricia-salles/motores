
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cd C:\Users\pati_\Documents\Motor\motores\Logistic_Regression

df = pd.read_csv('Social_Network_Ads.csv')
x = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.25, random_state = 0)

# Feature Scaling - padronização dos valores 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)  ## garantir o mesmo resultado
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  ## (valores reais, valores preditos pelo modelo)
### 65 e 24 são os valores preditos corretamente e 3 e 8 predições incorretas


# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''
pontos vermelhos = 0 (não compra SUV) e  pontos verdes = 1 (compra SUV)
região vermelha e região verde (modelo)
no geral, pessoas mais jovens e com baixo salário não compra, ao contrário dos 
mais velhos e com salário maior
'''
# Visualising the Test set results
x_set, y_set = x_test, y_test
##prepara o tamanho do gráfico que será plotado os valores/pontos, eixos x e y
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
'''próximo código vai pintar todos os pontos verdes e vermelhos no gráfico, 
vai acontecer o processo de classificação e será plotado a 'decision boundary'
'''
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
'''plotar todos os pontos preditos no gráfico
'''
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''
nesse gráfico aparece os 11 pontos que foram preditos incorretamente, como
foi visto na matriz de confusão