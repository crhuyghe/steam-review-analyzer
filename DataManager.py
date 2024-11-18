# Created by Christian Huyghe on 11/17/2024
# Manages review data
import json
import os
import steamreviews
import numpy as np
import pandas as pd
from time import sleep

from sentence_transformers import SentenceTransformer

class DataManager:
    def __init__(self):
        self.curr_game = None
        self.curr_app_id = None
        self.known_games = {}  # {app_id: "game_title"}

        # Scanning data file
        if os.path.exists("data/DataManager.json"):
            with open("data/DataManager.json") as f:
                data = json.load(f)
                for appid in data.keys():
                    if os.path.exists(f"data/embedding_{appid}.csv") and os.path.exists(f"data/trimmed_review_{appid}.csv"):
                        self.known_games[appid] = data[appid]
        else:
            json.dump({}, open("data/DataManager.json", "w"))

        self.review_df = None
        self.embeddings = None
        self.pop_score = None
        self.query_response = None
        self._pos_indices = None
        self._neg_indices = None

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_game(self, title: str, app_id: str, status_func):
        """Loads in reviews and embeddings for the selected game app id"""
        self.pop_score = None
        self.query_response = None
        if len(title) == 0 or not app_id.isdigit() or int(app_id) % 10 != 0:
            self.review_df = None
            self.embeddings = None
            self.curr_game = None
            self.curr_app_id = None
            status_func("Invalid Title or App ID")
            sleep(1)
            status_func("Idle")
            return
        if app_id in self.known_games.keys():
            status_func(f"Reloading game {app_id}...")
            self.curr_app_id = app_id
            self.curr_game = title
            if title != self.known_games[app_id]:
                self.known_games[app_id] = title

            self.review_df = pd.read_csv(f"data/trimmed_review_{app_id}.csv",
                                         dtype={"review": "string", "rating": "boolean"})
            self.embeddings = pd.read_csv(f"data/embedding_{app_id}.csv", header=None).to_numpy()

        else:
            self.curr_app_id = app_id
            self.curr_game = title

            if os.path.exists(f"data/embedding_{app_id}.csv"):
                status_func(f"Registering game {app_id}...")
                self.review_df = pd.read_csv(f"data/trimmed_review_{app_id}.csv",
                                             dtype={"review": "string", "rating": "boolean"})
                self.embeddings = pd.read_csv(f"data/embedding_{app_id}.csv", header=None).to_numpy()
                self._pos_indices = self.review_df["rating"]
                self._neg_indices = ~self.review_df["rating"]

            else:
                if not os.path.exists(f"data/trimmed_review_{app_id}.csv"):
                    if not os.path.exists(f"data/review_{app_id}.json"):
                        status_func(f"Downloading reviews for game {app_id}...")
                        self._download_reviews(app_id)
                    status_func(f"Trimming reviews for game {app_id}...")
                    self.review_df = self._trim_reviews(app_id)
                    os.remove(f"data/review_{app_id}.json")
                    if self.review_df is None:
                        self.embeddings = None
                        self.curr_game = None
                        self.curr_app_id = None
                        status_func(f"No reviews found for game {app_id}...")
                        sleep(1)
                        status_func("Idle")
                        return
                else:
                    self.review_df = pd.read_csv(f"data/trimmed_review_{app_id}.csv",
                                                 dtype={"review": "string", "rating": "boolean"})

                status_func(f"Creating review embeddings for game {app_id}...")
                pos_indices = self.review_df["rating"]
                neg_indices = ~self.review_df["rating"]

                self.embeddings = self.embedding_model.encode(self.review_df["review"].to_numpy())

                status_func(f"Cleaning reviews for game {app_id}...")

                pos_mean = np.mean(self.embeddings[pos_indices], axis=0)
                pos_mean /= np.linalg.norm(pos_mean)
                neg_mean = np.mean(self.embeddings[neg_indices], axis=0)
                neg_mean /= np.linalg.norm(neg_mean)

                temp = self.review_df.assign(sim=0.0)

                temp.loc[temp["rating"], "sim"] = np.dot(self.embeddings[pos_indices], pos_mean)
                temp.loc[~temp["rating"], "sim"] = np.dot(self.embeddings[neg_indices], neg_mean)

                self.review_df = self.review_df.loc[temp["sim"] > .05]
                self.embeddings = self.embeddings[temp["sim"] > .05]

                self.review_df.to_csv(f"data/trimmed_review_{app_id}.csv", index=False)
                pd.DataFrame(self.embeddings).to_csv(f"data/embedding_{app_id}.csv", index=False, header=False)

                self._pos_indices = self.review_df["rating"]
                self._neg_indices = ~self.review_df["rating"]

            self.known_games[app_id] = title
            json.dump(self.known_games, open("data/DataManager.json", "w"))
        self.pop_score = self._get_rating()
        status_func("Idle")

    def query(self, query):
        if self.review_df is None:
            return

        embed_query = self.embedding_model.encode(query)
        self.review_df["similarity"] = np.dot(embed_query, self.embeddings.T)
        mean_pos = np.mean(self.review_df.loc[self.review_df["rating"], "similarity"])
        mean_neg = np.mean(self.review_df.loc[~self.review_df["rating"], "similarity"])
        sorted_df = self.review_df.loc[self.review_df["similarity"] > .4].sort_values(by=["similarity"], ascending=False)

        positive = sorted_df.loc[sorted_df["rating"], "review"].values.tolist()
        negative = sorted_df.loc[~sorted_df["rating"], "review"].values.tolist()

        sentiment = f"{mean_pos * 100 / (mean_neg + mean_pos):.1f}% Positive"
        relevance = f"{np.mean(self.review_df["similarity"]) * 100:.1f}% Relevant"
        relevant_reviews = {"positive": positive, "negative": negative}

        self.query_response = (sentiment, relevance, relevant_reviews)

    def _get_rating(self):
        """Returns the ratio of positive and negative reviews"""
        if self.review_df is not None:
            return f"{len(self.review_df.loc[self.review_df["rating"]]) * 100 / len(self.review_df):.1f}% Positive"

    @staticmethod
    def _download_reviews(app_id):
        steamreviews.download_reviews_for_app_id(app_id,
                                                 chosen_request_params={"language": "english"})

    @staticmethod
    def _trim_reviews(app_id) -> pd.DataFrame | None:
        review_json = json.load(open(f"data/review_{app_id}.json"))
        reviews = pd.DataFrame(
            [review_json["reviews"][key]["review"] for key in review_json["reviews"].keys()],
            columns=["review"], dtype="string")
        if len(reviews) == 0:
            return None
        ratings = pd.DataFrame(
            [review_json["reviews"][key]["voted_up"] for key in review_json["reviews"].keys()],
            columns=["rating"], dtype="boolean")
        df = pd.concat([reviews, ratings], axis="columns")
        df.to_csv(f"data/trimmed_review_{app_id}.csv", index=False)

        df = pd.read_csv(f"data/trimmed_review_{app_id}.csv", dtype={"review": "string", "rating": "boolean"}).dropna().loc[~(df["review"].str.contains("(?:[^\x00-\x7F].*?){6}", regex=True))]
        df.to_csv(f"data/trimmed_review_{app_id}.csv", index=False)
        return df
