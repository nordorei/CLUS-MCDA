from sklearn.cluster import KMeans
import util.readData as dataProvider

businessAreas = dataProvider.getBusinessAreasList()

suppliersData = {}
for area in businessAreas.values():
    suppliersData[area] = dataProvider.getSuppliersDataFromBusinessArea(area)

k = 5 #default number of clusters
kMeans = KMeans(n_clusters=k)