import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("all-mpnet-base-v2")

# App ID for Astroneer
curr_app_id = 361420

if not os.path.exists(f"embedding_{curr_app_id}.csv"):
    if os.path.exists(f"trimmed_review_{curr_app_id}.csv"):
        df = pd.read_csv(f"trimmed_review_{curr_app_id}.csv", dtype={"review": "string", "rating": "boolean"}).dropna()
        reviews = df.iloc[:, :1]
        ratings = df.iloc[:, 1:]
    else:
        if os.path.exists(f"review_{curr_app_id}.json"):
            review_json = json.load(open(f"review_{curr_app_id}.json"))
            reviews = pd.DataFrame([review_json["reviews"][key]["review"] for key in review_json["reviews"].keys()],
                                   columns=["review"], dtype="string")
            ratings = pd.DataFrame([review_json["reviews"][key]["voted_up"] for key in review_json["reviews"].keys()],
                                   columns=["rating"], dtype="boolean")
            df = pd.concat([reviews, ratings], axis="columns")
            df.to_csv(f"trimmed_review_{curr_app_id}.csv", index=False)

            # Reloading and cleaning review data
            df = pd.read_csv(f"trimmed_review_{curr_app_id}.csv", dtype={"review": "string", "rating": "boolean"}).dropna().loc[~(df["review"].str.contains("(?:[^\x00-\x7F].*?){6}", regex=True))]
            df.to_csv(f"trimmed_review_{curr_app_id}.csv", index=False)
        else:
            raise SystemExit(f"Reviews for game {curr_app_id} not found.")

    embeddings = pd.DataFrame(embedding_model.encode(reviews.to_numpy().T[0]))
    embeddings.to_csv(f"embedding_{curr_app_id}.csv", index=False, header=False)
else:
    embeddings = pd.read_csv(f"embedding_{curr_app_id}.csv", header=None).to_numpy()
    df = pd.read_csv(f"trimmed_review_{curr_app_id}.csv", dtype={"review": "string", "rating": "boolean"})

    pos_reviews = df.loc[df["rating"]]
    pos_embeddings = embeddings[df["rating"]]

    neg_reviews = df.loc[~df["rating"]]
    neg_embeddings = embeddings[~df["rating"]]

    # pos_mean = np.mean(pos_embeddings, axis=0)
    # pos_mean /= np.linalg.norm(pos_mean)
    # neg_mean = np.mean(neg_embeddings, axis=0)
    # neg_mean /= np.linalg.norm(neg_mean)

    # new_df = df.assign(sim=np.zeros(len(df)))
    # new_df.loc[df["rating"], "sim"] = np.dot(pos_embeddings, pos_mean)
    # new_df.loc[~df["rating"], "sim"] = np.dot(neg_embeddings, neg_mean)

    # print(len(df))
    # df = df.loc[new_df["sim"] > .05]
    # print(len(df))
    # print(new_df.loc[~new_df["rating"]])
    # similarities = np.dot(embeddings[df["rating"]], pos_mean)
    # best_indices = np.argsort(similarities)[::-1]
    # print(similarities[best_indices][-100:])

    # trimmed_pos_reviews = pos_reviews.iloc[best_indices].iloc[:-75]
    # trimmed_pos_embeddings = pos_embeddings[best_indices][:-75]

    # pca = PCA(2, random_state=20)
    # pca.fit(embeddings)
    # pos_X = pca.transform(pos_embeddings)
    # neg_X = pca.transform(neg_embeddings)
    # plt.scatter(pos_X[:, 0], pos_X[:, 1])
    # plt.scatter(neg_X[:, 0], neg_X[:, 1])
    # plt.title("2-component PCA of text data vectors")
    # plt.show()
    #
    # tsne = TSNE(2, random_state=20)
    # X = tsne.fit_transform(embeddings)
    # pos_X = X[df["rating"]][best_indices]
    # neg_X = X[~df["rating"]]
    # plt.scatter(pos_X[:, 0], pos_X[:, 1])
    # plt.scatter(neg_X[:, 0], neg_X[:, 1])
    # plt.title("2-component t-SNE of text data vectors")
    # plt.show()

    # mds = MDS(2, random_state=20)
    # X = mds.fit_transform(embeddings[::25])
    # pos_X = X[df.iloc[::25]["rating"]]
    # neg_X = X[~df.iloc[::25]["rating"]]
    # plt.scatter(pos_X[:, 0], pos_X[:, 1])
    # plt.scatter(neg_X[:, 0], neg_X[:, 1])
    # plt.title("2-component MDS of text data vectors")
    # plt.show()

    # plt.bar(["Positive", "Negative"], [len(pos_embeddings), len(neg_embeddings)])
    # plt.title("Amount of reviews")
    # plt.xlabel("Rating")
    # plt.ylabel("Number of reviews")
    # plt.show()

    query = ""
    while query != "stop":
        query = input("Enter review query: ")
        t = time.time()
        query_embedding = embedding_model.encode(query)
        df["similarity"] = np.dot(query_embedding, embeddings.T)
        p = np.sort(df.loc[df["rating"], "similarity"])[::-1]
        n = np.sort(df.loc[~df["rating"], "similarity"])[::-1]
        pm = np.mean(p)
        nm = np.mean(n)
        print(pm, nm)
        print(pm / (pm + nm))
        # print(np.mean(df["similarity"]))

        # print(df.iloc[np.argsort(df["similarity"])[::-1]])

        print(f"\n{time.time() - t} seconds")
