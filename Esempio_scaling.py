from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Valuta i punti più vicini
from sklearn.metrics import accuracy_score

dataset = load_wine()

# SENZA SCALING #

X = dataset['data']
y = dataset['target']

model = KNeighborsClassifier()
model.fit(X, y)  # Manca train test split
p = model.predict(X)

acc_non_scalati = accuracy_score(y, p)*100
print(f"accuracy non scalate %.2f%%" %acc_non_scalati)

# CON SCALING #

X = dataset['data']
y = dataset['target']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model2 = KNeighborsClassifier()
model2.fit(X, y)
p2 = model2.predict(X)

acc_scalati = accuracy_score(y, p2)*100
print(f"accuracy scalate %.2f%%" %acc_scalati)
