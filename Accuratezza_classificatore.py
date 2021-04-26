import numpy as np
import scikitplot as skplt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

def randomize(v, lab, prob = 0.2):
    v2 = []
    for el in v:
        if np.random.random() > prob:
            v2.append(el)
        else:
            v2.append(np.random.choice(lab))
    return v2

labels = ['cronaca', 'politica', 'sport']
y = np.random.choice(labels, 1000)  # Estrae 1000 valori a caso tra le label --> target
p = randomize(y, labels)  # Previsione randomica (precisione dell'80% circa) (senza un modello ML) solo a scopo illustrativo

acc = accuracy_score(y, p)*100
print("Accuracy: %.2f%%" % acc)
print("Misclassification: %.2f%%" % (100-acc))

report = classification_report(y, p)
print(f"Report: \n {report}")

# Matrice di confusione --> Fa vedere le "confusioni" del modello (quello che prevede vs il vero target)
skplt.metrics.plot_confusion_matrix(y, p)
plt.show()
