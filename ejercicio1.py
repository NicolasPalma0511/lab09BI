import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'BI_Postulantes09.xlsx'  
df = pd.read_excel(file_path, sheet_name='Hoja1')

data_numeric = df[['Apertura Nuevos Conoc.', 'Nivel Organización', 'Participación Grupo Social', 
                   'Grado Empatía', 'Grado Nerviosismo', 'Dependencia Internet']]

kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(data_numeric)

centroids = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Nom_Especialidad', hue='Cluster', multiple='stack', shrink=0.8)
plt.title('Distribución de Conglomerados por Especialidad')
plt.xticks(rotation=45)
plt.show()
