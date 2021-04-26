from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(2)  # Imposto il seed di randomizzazione a 2 --> Ottengo gli stessi risultati sempre

dataset = load_boston()

X = dataset['data']  # features
y = dataset['target']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y)  # Divido i feature e i target in due parti (una per
# training, una per test)

model = LinearRegression()
model.fit(X_train, y_train)  # addestra il modello con i dati di training

p_train = model.predict(X_train)
p_test = model.predict(X_test)  # previsioni

mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)  # misura gli errori tra risposte desiderate e predizioni
print("MAE test", mae_test)  # Errore medio assoluto
print("MAE train", mae_train)
print(np.mean(y_test))  # media target


