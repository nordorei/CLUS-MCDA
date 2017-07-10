from sklearn.cluster import KMeans
from matplotlib import pyplot
from scipy.spatial import ConvexHull
import util.readData as dataProvider
import numpy as np

businessAreas = dataProvider.getBusinessAreasList()
k = 5 # default number of clusters
suppliersData = {}
kMeans = KMeans(n_clusters=k)

def plotUnitsPerBusinessAreas(suppliersData):
    plot_handles = []
    for area in suppliersData:
        ds = [item for item in suppliersData[area].values()]
        X = [item[0] for item in ds]
        Y = [item[1] for item in ds]
        plot, = pyplot.plot(X,Y,'o', label=area)
        plot_handles.append(plot)

    #pyplot.legend(handles=plot_handles, bbox_to_anchor=(1.05, 2.5), loc=2, borderaxespad=0.)
    pyplot.show()
    return None

def __initClustering(clusters):
    """
    """
    suppliersData = {}
    for area in businessAreas.values():
        suppliersData[area] = dataProvider.getSuppliersDataFromBusinessArea(area)
    
    k = clusters
    kMeans = KMeans(n_clusters=k)
    
    mustBeInClustering = {}
    for area in businessAreas.values():
        mustBeInClustering[area] = False

    return suppliersData, k, kMeans, mustBeInClustering

def runKMeans(clusters=5):
    """
    """
    suppliersData, k, kMeans, mustBeInClustering = __initClustering(clusters)

    for area in businessAreas.values():
        data = []
        for item in suppliersData[area].values():
            data.append(np.array([item[0], item[1]]))

        if len(data) <= 5: # no need for clustering
            print("BLAH")
            continue

        data = np.array(data)
        kMeans.fit(data)
        labels = kMeans.labels_
        centroids = kMeans.cluster_centers_

        for i in range(k):
            # select only data observations with cluster label == i
            ds = data[np.where(labels==i)]
            # plot the data observations
            pyplot.plot(ds[:,0],ds[:,1],'o')
            # plot the centroids
            lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
            # make the centroid x's bigger
            pyplot.setp(lines,ms=15.0)
            pyplot.setp(lines,mew=2.0)
        pyplot.show()

def runKMeansForAll(clusters=5):
    """
    """
    suppliersData, k, kMeans, mustBeInClustering = __initClustering(clusters)

    for area in businessAreas.values():
        data = []
        for item in suppliersData[area].values():
            data.append(np.array([item[0], item[1]]))

        if len(data) <= 5: # no need for clustering
            print("BLAH")
            continue

        data = np.array(data)
        kMeans.fit(data)
        labels = kMeans.labels_
        centroids = kMeans.cluster_centers_

        for i in range(k):
            # select only data observations with cluster label == i
            ds = data[np.where(labels==i)]
            # plot the data observations
            pyplot.plot(ds[:,0],ds[:,1],'o')
            # plot the centroids
            lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
            # make the centroid x's bigger
            pyplot.setp(lines,ms=15.0)
            pyplot.setp(lines,mew=2.0)

            points = np.array([np.array([row[0], row[1]]) for row in ds])
            if points.shape[0] < 3:
                continue
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                pyplot.plot(points[simplex, 0], points[simplex, 1], 'k-')
            
            pyplot.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
            pyplot.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

        pyplot.show()


if __name__ == '__main__':
    runKMeansForAll()