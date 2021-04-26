from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

n = 100
X = np.random.random(size=(n, 5))  # Crea matrice di feature randomica
y = np.random.choice(['si', 'no'], size=n)  # Crea matrice di target con si o no


X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPClassifier(hidden_layer_sizes=[1000, 500])  #Crea un Neural Network con 2 hidden layer (uno da 1000 nodi e l'altro da 500)
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train {acc_train}, Test {acc_test}')