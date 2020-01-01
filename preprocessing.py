import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

"""
Preprocessing 
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Overture Base de données et séparation Xi et Yi
data = pd.read_csv('./villes.csv', sep=';')
# .ix Permet d'accéder aux indices respectives pour les lignes et les colonnes (Dernier Element Inclus)
# Possible d'utiliser iloc
X = data.ix[:, 1:13].values
labels = data.ix[:, 0].values

"""
Remplacement des valeurs manquantes 
"""

from sklearn.preprocessing import Imputer
X_num = np.copy(X_n)
X_num[X_num == '?'] = np.nan
X_num = X_num.astype(float)
imp_num = Imputer(missing_values=np.nan, strategy='mean')
X_num = imp_num.fit_transform(X_num)

"""
Encodage des variables Catégorielle
"""

from sklearn.preprocessing import OneHotEncoder
X_cat_bin = OneHotEncoder().fit_transform(X_cat).toarray()

"""
Normalisation des données 
"""

from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
SS.fit(X)
X_norm = SS.transform(X)

"""
ACP
"""

from sklearn.processing import PCA

pca = PCA(n_components = 0.90) # PCA tel que 90% de la variance soit expliquée 
pca.fit(X)
X_pca = pca.transform(X_fit)
# Pour chaque prédicteur pca affiche un vecteur tel que chaque composante présent le coeficient de corrélation avec l'ancien prédicteur à l'indice i
coeffs = pca.components_



