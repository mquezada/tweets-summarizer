from sklearn import metrics
from sklearn.metrics import pairwise_distances
from itertools import islice
from word2vec import km, model, urls, labels
#La función de distancia es euclidiana.

#Calcular el valor de silhoutte para todos los clusters
def get_silhoutte(model, labels):
    return metrics.silhouette_score(model, labels)

# Dado el número de un cluster, las etiquetas de los elementos y los modelos
# entrega una lista con todos los elementos que pertenecen a ese cluster
def get_clusters_elements(cluster_number, labels, model):
    indexes= [i for i, j in enumerate(labels) if j == cluster_number]
    return [model[index] for index in indexes]

#Dado los elementos de un cluster, calcula su diametro, i.e, la distancia entre los 2 puntos más lejanos
def get_cluster_diameter(cluster):
    distance=pairwise_distances(cluster)
    return max(map(lambda x: x[len(distance)-1], distance))

#Entrega una lista con las distancias entre los centros de cada cluster
def get_distances_centers(centers):
    return pairwise_distances(centers)

#Entrega la distancia min o max (espeficicado por function) de todos los clusters con respecto a todos los otros clusters
def get_distances(labels, model, number_cluster, function):
    distances_all=[]
    for i in range(number_cluster):
        elements_cluster_1=get_clusters_elements(i,labels,model)
        distances_clusters=[]
        for j in islice(range(number_cluster), i+1, None):
            elements_cluster_2=get_clusters_elements(j,labels,model)
            distance=pairwise_distances(elements_cluster_1,elements_cluster_2)
            value=function(map(lambda x: x[len(distance[0])-1], distance))
            distances_clusters.append(value)
        distances_all.append(distances_clusters)
    return distances_all

print(get_distances(labels,model[urls],km.n_clusters,max))

        

