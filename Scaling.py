# Motivi per lo scaling:
# 1. Equalizza l'impatto delle features (Differenze di varianza delle feature ridotte) --> Compattazione delle feature
# 2. Learning più veloce

from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


dataset = load_wine()

X = dataset['data'][:, [4,7]]  # Prendo solo la colonna del MG e dei fenoli (hanno molta varianza)

"""
# SENZA SCALING #
df = pd.DataFrame(X, columns=['magnesium', 'phenols'])
g = sns.scatterplot(data=df, x='magnesium', y='phenols')  # Crea un grafico
g.set(xlim=(-10,200), ylim=(-10,200))  # Imposta una scala per gli assi (se no in automatico usa la scala più adatta, ma non mostra la varianza delle feature)
# plt.show()
"""

# CON SCALING #
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

df = pd.DataFrame(X, columns=['magnesium', 'phenols'])
g = sns.scatterplot(data=df, x='magnesium', y='phenols')  # Crea un grafico
plt.show()