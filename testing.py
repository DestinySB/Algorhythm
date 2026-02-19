import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas 
import numpy 
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# AUTHENTICATION
# ---------------------------
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="http://localhost:8888/callback",
    scope="user-top-read"
))

# ---------------------------
# GET USER TOP TRACKS
# ---------------------------
results = sp.current_user_top_tracks(limit=20)

tracks = []
for item in results["items"]:
    tracks.append({
        "id": item["id"],
        "name": item["name"],
        "artist": item["artists"][0]["name"]
    })

# Convert to DataFrame
df = pandas.DataFrame(tracks)

# ---------------------------
# GET AUDIO FEATURES
# ---------------------------
features = sp.audio_features(df["id"].tolist())

feature_df = pandas.DataFrame(features)[[
    "danceability",
    "energy",
    "tempo",
    "valence"
]]

# Combine track info + features
df = pandas.concat([df, feature_df], axis=1)

# ---------------------------
# BUILD USER PROFILE VECTOR
# ---------------------------
user_profile = df[["danceability", "energy", "tempo", "valence"]].mean().values.reshape(1, -1)

# ---------------------------
# SEARCH FOR CANDIDATE SONGS
# ---------------------------
search_results = sp.search(q="genre:pop", type="track", limit=20)

candidate_tracks = []
for item in search_results["tracks"]["items"]:
    candidate_tracks.append({
        "id": item["id"],
        "name": item["name"],
        "artist": item["artists"][0]["name"]
    })

candidate_df = pandas.DataFrame(candidate_tracks)

candidate_features = sp.audio_features(candidate_df["id"].tolist())

candidate_feature_df = pandas.DataFrame(candidate_features)[[
    "danceability",
    "energy",
    "tempo",
    "valence"
]]

candidate_df = pandas.concat([candidate_df, candidate_feature_df], axis=1)

# ---------------------------
# COMPUTE SIMILARITY
# ---------------------------
similarities = cosine_similarity(
    user_profile,
    candidate_df[["danceability", "energy", "tempo", "valence"]]
)

candidate_df["similarity"] = similarities.flatten()

# ---------------------------
# RANK AND DISPLAY RESULTS
# ---------------------------
recommended = candidate_df.sort_values(by="similarity", ascending=False).head(5)

print("\nTop 5 Recommended Songs:\n")
for _, row in recommended.iterrows():
    print(f"{row['name']} by {row['artist']} (Score: {row['similarity']:.3f})")
