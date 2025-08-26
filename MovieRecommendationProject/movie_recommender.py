import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")

##Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director']

##Step 3: Create a column in DF which combines all selected features
    # Removing all NaN in features by filling it with an empty string
for feature in features:
	df[feature] = df[feature].fillna("")
	
def combine_features(row):
	return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]

df["combine_features"] = df.apply(combine_features, axis=1)

# print("Combined Features: ", df["combine_features"].head())

##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combine_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

    # Building a dictionary for movie titles
title_to_index = {}
for i, t in enumerate(df["title"].astype(str)):
	lower_title = t.lower()
	title_to_index[lower_title] = i

def user_input():
	while True:
		enter_input = input("Enter movie title OR press Enter to quit: ").strip()
		if enter_input == "":
			return None
		key = enter_input.lower()
		
		# Exact match
		if key in title_to_index:
			return title_to_index[key]

        # If there are no exact match, make suggestions
		list_titles = list(title_to_index.keys())
		suggestions = difflib.get_close_matches(key, list_titles, n = 5, cutoff=0.5)
		if suggestions:
			print("Could not find a match. Did you mean: ")
			for i, s in enumerate(suggestions, 1):
				print(f" {i}. {df['title'].iloc[title_to_index[s]]}")
			choices = input("Choose a number OR Enter to try again: ").strip()
			if choices.isdigit():
				ind = int(choices)
				if 1 <= ind <= len(suggestions):
					return title_to_index[suggestions[ind - 1]]
			else:
				print("No close matches found. Try again. \n")
	
def main():
	## Step 6: Get index of this movie from its title
    movie_index = user_input()
    if movie_index is None:
        print ("No movie selected")
        

    sim_movies = list(enumerate(cosine_sim[movie_index]))

    ## Step 7: Get a list of similar movies in descending order of similarity score
    sorted_sim_movies = sorted(sim_movies, key=lambda x:x[1], reverse=True)

    ## Step 8: Print titles of first 10 movies
    recommended = sorted_sim_movies[1:11]
    print(f"\n Top 10 similar movies to {get_title_from_index(movie_index)}")
    count = 0
    for movie in recommended:
        count += 1
        print(f"{count}. ", get_title_from_index(movie[0]))
        if count > 10:
            break

if __name__ == "__main__":
	main()