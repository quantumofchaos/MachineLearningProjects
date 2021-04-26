import numpy as np
from sklearn.compose import ColumnTransformer
from  sklearn.impute import SimpleImputer

X = [
    [20, np.nan],
    [np.nan, 'm'],
    [30, 'f'],
    [35, 'f'],
    [np.nan, np.nan]  #np.nan -> Not A Number (Valore mancante
]

transformers = [
    #['imputer', SimpleImputer(strategy='most_frequent'), [0, 1]]  #strategy = "most frequent" usa il termine più frequente per fillare il vaolre mancante
    ['age_imputer', SimpleImputer(strategy='median'), [0]]  #Usa una media per riempire i valori mancanti
    ['sex_imputer', SimpleImputer(strategy='constant', fill_value='n.d.'), [1]] #sostituisce i valori mancanti con una costante "n.d."
]
ct = ColumnTransformer(transformers)

X = ct.fit_transform(X)