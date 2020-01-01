"""
Méthodes de clustering
"""

# importation de l'objet KMeans
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_norm)

clustering = kmeans.labels_
colors = ['red','yellow','blue','pink']
plt.scatter(X_pca[:, 0], X_pca[:, 1], c= clustering, cmap=matplotlib.colors.ListedColormap(colors))
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()

# Clustering Agglomérative 

aggloClust = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_norm)

"""
Méthodes d'évaluation de clustering 
"""

# Silouhette score

from sklearn import metrics
for i in np.arange(2, 6):
    clustering = KMeans(n_clusters=i).fit_predict(X_norm)
    print(metrics.silhouette_score(X, clustering,metric='euclidean'))