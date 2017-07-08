from sklearn.cluster import KMeans
from matplotlib import pyplot
import util.readData as dataProvider
import numpy as np

businessAreas = dataProvider.getBusinessAreasList()

suppliersData = {}
for area in businessAreas.values():
    suppliersData[area] = dataProvider.getSuppliersDataFromBusinessArea(area)

k = 5 #default number of clusters
kMeans = KMeans(n_clusters=k)

#testing on first business area
data = []
for item in suppliersData['Electrical Tools'].values():
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