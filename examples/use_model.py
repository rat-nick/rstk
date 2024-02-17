from rstk.algo.content_based.knn import KNN

# load the model
model = KNN.deserialize(KNN, "model.pkl")

print(model.get_similar_items([1, 2], k=3))
