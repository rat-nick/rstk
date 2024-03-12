from rstk.model._knn import KNN

# load the model
model = KNN.deserialize("model.pkl")

print(model.get_similar_items([1, 2], k=3))
