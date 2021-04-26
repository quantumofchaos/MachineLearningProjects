from sklearn.feature_extraction.text import CountVectorizer

X = [
    'ciao ciao miao',
    'miao',
    'miao bao'
]

vectorizer = CountVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

print(vectorizer.get_feature_names())  # ['bao', 'ciao', 'miao']
print(X.todense())  # Matrice densa
print(X)  # Matrice sparsa 
