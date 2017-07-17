from sklearn.cluster import KMeans
from matplotlib import pyplot
from scipy.spatial import ConvexHull
import util.readData as dataProvider
import numpy as np
import math

businessAreas = dataProvider.getBusinessAreasList()
k = 5 # default number of clusters
sample_case_study = 'Contractor'
min_columns = [0 , 5 , 6]
max_columns = [1, 2, 3, 4]
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
        suppliersData[area] = dataProvider.getSuppliersData(area)
    
    k = clusters
    kMeans = KMeans(n_clusters=k)
    
    mustBeInClustering = {}
    for area in businessAreas.values():
        mustBeInClustering[area] = True

    return suppliersData, k, kMeans, mustBeInClustering


def __isClusteringNeeded(mustBeInClustering):
    """
    """
    for area in businessAreas.values():
        if mustBeInClustering[area]:
            return True

    return False


def runKMeansForAllAreas(suppliersData, k, kMeans, mustBeInClustering):
    """
    """
    clusters = {}
    for area in businessAreas.values():
        if mustBeInClustering[area] == False:
            continue # ready for ranking

        data = []
        for item in suppliersData[area]:
            costPerPrice = suppliersData[area][item][0]
            data.append(np.array([item, costPerPrice]))

        if len(data) <= 5: # no clustering needed
            mustBeInClustering[area] = False
            clusters[area] = {'FinalCandidates': np.array(data)}
            continue

        data = np.array(data)
        kMeans.fit(data)
        labels = kMeans.labels_
        centroids = kMeans.cluster_centers_

        plot_handles = []
        clusterData = {}
        for i in range(k):
            clusterLabel = 'Cluster{}'.format(i + 1)
            # select only data observations with cluster label == i
            ds = data[np.where(labels==i)]
            clusterData[clusterLabel] = ds[:, :2]
            # plot the data observations
            plot, = pyplot.plot(ds[:,0],ds[:,1],'o', label=clusterLabel)
            plot_handles.append(plot)
            # plot the centroids
            lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
            # make the centroid x's bigger
            pyplot.setp(lines,ms=15.0)
            pyplot.setp(lines,mew=2.0)

            # points = np.array([np.array([row[0], row[1]]) for row in ds])
            # if points.shape[0] < 3:
            #     continue
            # hull = ConvexHull(points)
            # for simplex in hull.simplices:
            #     pyplot.plot(points[simplex, 0], points[simplex, 1], 'k-')
            
            # pyplot.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
            # pyplot.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

        clusters[area] = clusterData
        # pyplot.legend(handles=plot_handles, bbox_to_anchor=(1.05, 2.5), loc=2, borderaxespad=0.)
        # pyplot.show()
        

    return clusters, mustBeInClustering

def runCLUSMCDA(k_clusters=5):
    """
    """
    suppliersData, k, kMeans, mustBeInClustering = __initClustering(k_clusters)

    #while __isClusteringNeeded(mustBeInClustering):
    clusters, mustBeInClustering = runKMeansForAllAreas(suppliersData, k, kMeans, mustBeInClustering)
    for area in businessAreas.values():
        areaData = []
        for cluster in clusters[area]:
            areaClusterRows = clusters[area][cluster][:,0]
            # print(area, cluster, areaClusterRows)

            x = np.array([dataProvider.getRow(row) for row in areaClusterRows])
            w = [0.207317073, 0.12195122, 0.170731707, 0.12195122, 0.097560976, 0.146341463, 0.134146341]
            n = len(w) # number of columns

            # normalizing X
            X = []
            for i in range(len(x)):
                row = []
                for j in range(len(x[i])):
                    RoSoS = 0 # Root of Sum of Squares
                    for k in range(len(x)):
                        RoSoS += (x[k][j]) ** 2
                    RoSoS = math.sqrt(RoSoS)

                    row.append(x[i][j] / RoSoS)
                X.append(np.array(row))
                    
            X = np.array(X)

            # calculating Y
            Y = []
            for i in range(len(X)):
                Yjg = 0
                for g in max_columns:
                    Yjg += w[g] * X[i][g]

                Ygn = 0
                for ng in min_columns:
                    Ygn += w[ng] * X[i][ng]

                Yi = Yjg - Ygn
                Y.append(Yi)

            Y = np.array(Y)

            # calculating R
            R = []
            for j in range(n):
                rj = X[0][j]
                if j in min_columns:
                    for i in range(len(X)):
                        if rj > X[i][j]:
                            rj = X[i][j]

                else:
                    for i in range(len(X)):
                        if rj < X[i][j]:
                            rj = X[i][j]

                R.append(rj)

            R = np.array(R)

            # calculating Z
            Z = []
            for i in range(len(X)):
                zi = X[i][0]
                for j in range(n):
                    exp = abs(w[j] * R[j] - w[j] * X[i][j])
                    if zi < exp:
                        zi = exp

                Z.append(zi)

            Z = np.array(Z)
            print(Z)
            


if __name__ == '__main__':
    runCLUSMCDA()