"""
Nota: Para el correcto funcionamiento del script
instale las siguientes librerias:
    ->numpy
    ->pandas
    ->sklearn
    
para instalarlas puede utilizar el comando pip install [LIBRERIA]
directamente desde la terminal.
"""

#LIBRERIAS A UTILIZAR
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#IMPORTANDO LOS DATOS

#Reemplazo la ruta por la ruta en la que tiene los archivos
#en su dispositivo
path_test = "C:\\Users\\odind\\OneDrive\\Escritorio\\test.csv"
path_train = "C:\\Users\\odind\\OneDrive\\Escritorio\\train.csv"

df_test = pd.read_csv(path_test)
df_train = pd.read_csv(path_train)

print(df_test.head())
print(df_train.head())

#ENTENDIMIENTO DE LA DATA
print("Cantidad de datos: ")
print(df_train.shape)
print(df_test.shape)

print("Tipos de datos: ")
print(df_train.info())
print(df_test.info())

print("Datos faltantes: ")
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print("Estadísticas del dataset: ")
print(df_train.describe())
print(df_test.describe())

#PREPROCESAMIENTO DE LA DATA

df_train["Sex"].replace(["female", "male"],[0,1], inplace=True)
df_test["Sex"].replace(["female", "male"], [0,1], inplace=True)

df_train["Embarked"].replace(['Q','S','C'],[0,1,2], inplace=True)
df_test["Embarked"].replace(['Q','S','C'],[0,1,2], inplace=True)

print(df_train["Age"].mean())
print(df_test["Age"].mean())
promedio = 30
df_train["Age"] = df_train["Age"].replace(np.nan, promedio)
df_test["Age"] = df_test["Age"].replace(np.nan, promedio)

df_train.drop(["Cabin"], axis = 1, inplace=True)
df_test.drop(["Cabin"], axis = 1, inplace=True)

df_train = df_train.drop(["PassengerId", "Name", "Ticket"], axis=1)
df_test = df_test.drop(["Name", "Ticket"], axis=1)

df_train.dropna(axis=0, how="any", inplace=True)
df_test.dropna(axis=0, how="any", inplace=True)

print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print(df_train.shape)
print(df_test.shape)

print(df_test.head())
print(df_train.head())

#APLICACIÓN DEL ALGORITMO DE MACHINE LEARNING

X = np.array(df_train.drop(["Survived"], axis=1))
y = np.array(df_train["Survived"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print("Presición Regresión Logística: ")
print(logreg.score(X_train, y_train))

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print("Precisión Soporte de Vectores: ")
print(svc.score(X_train, y_train))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print("Presición Vecinos más Cercanos: ")
print(knn.score(X_train, y_train))

#PREDICCIÓN UTILIZANDO LOS MODELOS

ids = df_test["PassengerId"]

prediccion_logreg = logreg.predict(df_test.drop("PassengerId", axis=1))
out_logreg = pd.DataFrame({"PassengerId": ids, "Survived": prediccion_logreg})
print("Predicción Regresión Logística: ")
print(out_logreg.head())

prediccion_svc = svc.predict(df_test.drop("PassengerId", axis=1))
out_svc = pd.DataFrame({"PassengerId": ids, "Survived": prediccion_svc})
print("Predicción Soporte de Vectores: ")
print(out_svc.head())

prediccion_knn = knn.predict(df_test.drop("PassengerId", axis=1))
out_knn = pd.DataFrame({"PassengerId": ids, "Survived": prediccion_knn})
print("Prediccion Vecinos más Cercanos: ")
print(out_knn.head())