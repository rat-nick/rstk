from rstk.preprocessor import Preprocessor

data = (
    Preprocessor()
    .load("dataset.csv")
    .handle_missing_values()
    .multilabel_binarize(["release", "genres"])
    .normalize(["price", "releaseYear"], methods=["z-score", "linear"])
    .select_features(regex=".*")
)

data.to_csv("prepreocessed.csv", index=False)
