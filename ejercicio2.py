import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel('BI_Clientes09.xlsx', sheet_name='Hoja1')

df['BirthDate'] = pd.to_datetime(df['BirthDate'])
df['Age'] = 2024 - df['BirthDate'].dt.year

df_clean = df[['GeographyKey', 'Age', 'MaritalStatus', 'YearOfFirstPurchase', 'CommuteDistance', 'BikeBuyer']].dropna()

df_clean['MaritalStatus'] = df_clean['MaritalStatus'].map({'M': 1, 'S': 0})
df_clean['CommuteDistance'] = df_clean['CommuteDistance'].map({'0-1 Miles': 0, '2-5 Miles': 1, '5-10 Miles': 2, '10+ Miles': 3})

X = df_clean[['GeographyKey', 'Age', 'MaritalStatus', 'YearOfFirstPurchase', 'CommuteDistance']]
y = df_clean['BikeBuyer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=3) 
clf.fit(X_train, y_train)

importances = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Bike', 'Bike'], rounded=True)
plt.title("Árbol de Decisiones para Predecir Compradores de Bicicleta")
plt.show()

for i, importance in enumerate(importances):
    print(f"Característica: {feature_names[i]} - Importancia: {importance:.4f}")
