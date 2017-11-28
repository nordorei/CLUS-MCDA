'''
This project is part of the CLUS-MCDA approach.

Corresponding author: Abteen Ijadi Maghsoodi
Algorithm Designer: Abteen Ijadi Maghsoodi
Software Developer: Azad Kavian
'''

from sklearn.cluster import KMeans
from matplotlib import pyplot
from scipy.spatial import ConvexHull
import util.readData as dataProvider
import numpy as np
import math

businessAreas = dataProvider.getBusinessAreasList().values()
sample_case_study = 'Contractor'
min_columns = [0 , 5 , 6]
max_columns = [1, 2, 3, 4]


def plotUnitsPerBusinessAreas(suppliersData):
    """
    ({Dict}) -> None

    Draws plots according to the suppliers data

    """
    plot_handles = []
    for area in suppliersData:
        X = [item[0] for item in suppliersData[area].values()]
        Y = [item[1] for item in suppliersData[area].values()]
        plot, = pyplot.plot(X,Y,'o', label=area)
        plot_handles.append(plot)

    # pyplot.legend(handles=plot_handles, bbox_to_anchor=(1.05, 2.5), loc=2, borderaxespad=0.)
    pyplot.show()
    return None


def __initClustering():
    """
    (None) -> {Dict}, {Dict}

    Initializes the two key dictionaries for the clustering.

    @returns
    suppliersData: a StringToDict dictionary. Receives a string as a key, representing the business area,
      gives a IntegerToList dictionary.
    mustBeInClustering: a StringToBoolean dictionary. Receives a string as a key, representing the business
      area, gives a boolean to determine if it must be in clustering cycle or not.

    """
    suppliersData = {}
    for area in businessAreas:
        suppliersData[area] = dataProvider.getSuppliersData(area)
    
    mustBeInClustering = {}
    for area in businessAreas:
        mustBeInClustering[area] = True

    return suppliersData, mustBeInClustering


def __isClusteringNeeded(mustBeInClustering):
    """
    ({Dict}) -> Boolean

    Checks if any business area must be in the clustering cycle or not.

    """
    for area in businessAreas:
        if mustBeInClustering[area]:
            return True

    return False


def runKMeansForAllAreas(suppliersData, k, mustBeInClustering):
    """
    ({Dict}, int, {Dict}) -> {Dict}, {Dict}

    Runs KMeans clustering method for all business areas and gives the clusters for each area in a dictionary.

    @returns
    clusters: a StringToDict dictionary. Receives a string as a key, representing the business area, gives a
      StringToList dictionary. 
    mustBeInClustering: a StringToBoolean dictionary. Receives a string as a key, representing the business
      area, gives a boolean to determine if it must be in clustering cycle or not.

    """
    clusters = {}
    for area in businessAreas:
        data = []
        for item in suppliersData[area]:
            costPerPrice = suppliersData[area][item][0]
            data.append(np.array([item, costPerPrice]))

        if len(data) <= 5: # no clustering needed
            mustBeInClustering[area] = False
            clusterData = {}
            for i in range(len(data)):
                clusterLabel = 'Candidate{}'.format(i + 1)
                clusterData[clusterLabel] = np.array([data[i]])
            clusters[area] = clusterData

        if mustBeInClustering[area] == False:
            continue # ready for ranking

        data = np.array(data)
        kMeans = KMeans(n_clusters=k)
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
            # plot, = pyplot.plot(ds[:,0],ds[:,1],'o', label=clusterLabel)
            # plot_handles.append(plot)
            # plot the centroids
            # lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
            # make the centroid x's bigger
            # pyplot.setp(lines,ms=15.0)
            # pyplot.setp(lines,mew=2.0)

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


def __getRanks(dataColumn, descending=False):
    """
    ([List], Boolean) -> [List]

    Determines the indexes(ranks) of data stored in the given list after getting sorted. Has an option to
      make it a descending ranking.

    """
    items = [item for item in dataColumn]
    itemsSorted = sorted(items, reverse=descending)

    ranks = [itemsSorted.index(item) + 1 for item in items]
    return ranks


def __getFinalRanks(yRanks, zRanks, uRanks):
    """
    ([List], [List], [List]) -> [List]

    Determines the final ranking of data rows by getting average of Y,Z and U rankings.

    """
    rankAverages = [(yRanks[i] + zRanks[i] + uRanks[i]) / 3 for i in range(len(yRanks))]
    sortedAverages = sorted(rankAverages)

    fRanks = []
    for item in rankAverages:
        rank = sortedAverages.index(item) + 1
        while rank in fRanks:
            rank += 1
        fRanks.append(rank)

    return fRanks


# This is where the magic happens ...
def runCLUSMCDA(k_clusters=5):
    """
    (int) -> None

    Runs the CLUS-MCDA algorithm on the given number of clusters.
    The default number is 5.

    """
    # Initializing the data ...
    suppliersData, mustBeInClustering = __initClustering()
    cycle = 0
    topClusters = {}

    # Running a loop until every cluster is done ...
    while __isClusteringNeeded(mustBeInClustering):
        cycle += 1
        # Running KMeans algorithim to get clusters ...
        clusters, mustBeInClustering = runKMeansForAllAreas(suppliersData, k_clusters, mustBeInClustering)
        rowsToBeRemoved = []
        topClusters = {}
        # Investigating each business area ...
        for area in businessAreas:
            areaClusterData = []            
            for cluster in clusters[area]:
                areaClusterRows = clusters[area][cluster][:,0]

                x = np.array([dataProvider.getRow(row) for row in areaClusterRows]) # Reading data from .xls sheet
                w = [0.207317073, 0.12195122, 0.170731707, 0.12195122, 0.097560976, 0.146341463, 0.134146341] # weights determined by the author
                n = len(w) # number of columns

                # normalizing X
                X = []
                for i in range(len(x)):
                    row = []
                    for j in range(len(x[i])):
                        RoSoS = 0.0 # Root of Sum of Squares
                        for k in range(len(x)):
                            RoSoS += float(x[k][j]) ** 2
                        RoSoS = float(math.sqrt(RoSoS))
                        if RoSoS == 0.0:
                            RoSoS = 0.00000001 # a tiny number
                        row.append(float(x[i][j]) / RoSoS)
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
                
                # calculating U
                U = []
                for i in range(len(X)):
                    up = 1
                    for g in max_columns:
                        up *= X[i][j] ** w[j]

                    bot = 1
                    for ng in min_columns:
                        bot *= X[i][j] ** w[j]

                    if bot == 0.0:
                        bot == 0.0000001 # a tiny number
                    ui = up / bot
                    U.append(ui)

                U = np.array(U)
                
                # gettting means
                y = 0
                z = 0
                u = 0
                no = len(Z)
                for i in range(no):
                    y += Y[i]
                    z += Z[i]
                    u += U[i]
                y /= no
                z /= no
                u /= no

                areaClusterData.append(np.array([cluster, sum(areaClusterRows), float(y), float(z), float(u)]))

            areaClusterData = np.array(areaClusterData)

            # determining rankings for each column
            yRanks = __getRanks(areaClusterData[:,2], descending=True)
            zRanks = __getRanks(areaClusterData[:,3])
            uRanks = __getRanks(areaClusterData[:,4], descending=True)

            fRanks = __getFinalRanks(yRanks, zRanks, uRanks)
            # appending ranks to data rows
            ranks = np.array([[yRanks[i], zRanks[i], uRanks[i], fRanks[i]] for i in range(len(yRanks))])
            
            areaClusterDataRanks = []
            for i in range(len(ranks)):
                oldRow = areaClusterData[i]
                newRow = np.insert(oldRow, len(oldRow), ranks[i])
                areaClusterDataRanks.append(newRow)

            areaClusterDataRanks = np.array(areaClusterDataRanks)

            # if area == sample_case_study:
            #     print('\nCycle', cycle, '\n')
            #     print('data in', area, ':', len(suppliersData[area]))
            #     print(areaClusterDataRanks)

            # finding top 3 clusters/candidates
            lowClusters = []
            for row in areaClusterDataRanks:
                rank = int(row[-1])
                cluster = row[0]
                if rank > 3:
                    lowClusters.append(cluster)
                else:
                    if 'Candidate' in cluster:
                        if area in topClusters:
                            topClusters[area][rank] = row[1]
                        else:
                            topClusters[area] = {rank: row[1]}

            # removing low clusters from data set
            for cluster in lowClusters:
                rowsToBeRemoved.extend(clusters[area][cluster][:,0].astype(int).tolist())


        rowsToBeRemoved.sort()
        for area in suppliersData:
            for uselessRow in rowsToBeRemoved:
                suppliersData[area].pop(uselessRow, None)

    # printing the final results
    for area in businessAreas:
        print(area)
        for rank in topClusters[area]:
            print('Rank {}: '.format(rank), topClusters[area][rank])
        print('\n')


if __name__ == '__main__':
    runCLUSMCDA()