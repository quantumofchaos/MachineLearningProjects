from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.arange(0, 10, 0.2)
n = len(X)
y = np.cos(X) + (2*np.random.random(n))  # Creo io un modello (relazione tra X e y)
# cos + "rumore"

# Converto l'array di numpy in un array leggibile da sklearn
X = np.expand_dims(X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPRegressor(hidden_layer_sizes=[50], max_iter=10000, tol=-1, verbose=2)  # Max iter --> Quanti "giri" fa il NN, verbose --> Printa l'output
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)
p = model.predict(X)

mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)
print(f'Train {mae_train}, Test {mae_test}')

sns.scatterplot(x=X_train[:,0], y=y_train)  # Apprendimento (BLU)
sns.scatterplot(x=X_test[:,0], y=y_test)  # Test (ARANCIONE)
sns.lineplot(x=X[:,0], y=p)  # Modello (linea)
plt.show()
