import pandas as pd

from rstk.algo.content_based.knn import KNN

# read the data into a dataframe
df = pd.read_csv("dataset.csv", delimiter="|").set_index("movie id")

# initialize the model
model = KNN(df, feature_columns=df.columns[5:])

# get some movie titles that are in the crime genre
items = [
    "Godfather",
    "Goodfellas",
    "Raging Bull",
    "Reservoir Dogs",
]

# find the ids of the rows that contain one of the following phrases in their title
ids = model.data.loc[df["movie title"].str.contains("|".join(items))].index.to_list()

# get recommendations for these items
recommendations = model.get_recommendations(preference=ids, k=10)

# convert the recommendations to a list of movie titles
recommendations = model.data.loc[recommendations]["movie title"].to_list()

# print the recommendations
print("Recommendations:")
[print(r) for r in recommendations]

# get some lightheart comedy movies
items = [
    "Sleepless in Seattle",
    "Naked Gun",
    "Beverly Hills Cop",
    "Three Musketeers",
]

# find the ids of the rows that contain one of the following phrases in their title
ids = model.data.loc[df["movie title"].str.contains("|".join(items))].index.to_list()

# get recommendations for these items
recommendations = model.get_recommendations(preference=ids, k=10)

# convert the recommendations to a list of movie titles
recommendations = model.data.loc[recommendations]["movie title"].to_list()

print("Recommendations:")
[print(r) for r in recommendations]

model.serialize("models/knn.pkl")
