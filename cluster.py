from sklearn.cluster import KMeans
from matplotlib import pyplot
import util.readData as dataProvider
import numpy as np

businessAreas = dataProvider.getBusinessAreasList()

suppliersData = {}
for area in businessAreas.values():
    suppliersData[area] = dataProvider.getSuppliersDataFromBusinessArea(area)

# plotting units in each business area
#pyplot.subplot(223)
plot_handles = []
for area in suppliersData:
    ds = [item for item in suppliersData[area].values()]
    X = [item[0] for item in ds]
    Y = [item[1] for item in ds]
    plot, = pyplot.plot(X,Y,'o', label=area)
    plot_handles.append(plot)

#pyplot.legend(handles=plot_handles, bbox_to_anchor=(1.05, 2.5), loc=2, borderaxespad=0.)
pyplot.show()

k = 5 #default number of clusters
kMeans = KMeans(n_clusters=k)

for area in businessAreas.values():
    data = []
    for item in suppliersData[area].values():
        data.append(np.array([item[0], item[1]]))

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