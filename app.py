from flask import Flask, request
import numpy as np
import pandas as pd
import re
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

### Function that tokenize and stem a text
def tokenize_and_stem(text):
    
    # We need to tokenize to perform the cosine distance analisys. We also stem the text to transform the words into their root form, so # words like "run" and "running" are grouped in the same token.
    # Create an English language SnowballStemmer object
    stemmer_en = SnowballStemmer("english")
    
    # Tokenize by sentence, then by word
    tokens = [ word for sent in sent_tokenize(text) for word in word_tokenize(sent) ]
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z0-9]', token)]
    
    # Stem the filtered_tokens
    stems = [ stemmer_en.stem(ft) for ft in filtered_tokens ]
    
    return stems

app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/", methods=['POST', 'GET'])
def home():

    movie_name_given_by_user = request.args.get('movie')
    if movie_name_given_by_user == None:
        return "Nome do filme não informado."
    
    #return "Movie: " + movie_name_given_by_user
    
    search_url = 'https://www.imdb.com/find'
    payload = {'q': movie_name_given_by_user}
    soup_search = BeautifulSoup(requests.get(search_url, params=payload).text, 'html.parser')
    movie_found = soup_search.select("a[href*='/title/tt']")

    if movie_found == []:
        return f'Filme \"{movie_name_given_by_user}\" não encontrado no Imdb. Verifique e tente novamente.'

    # Assumes that the first result of the imdb search is the correct one
    imdb_id_given_by_user = str.replace(movie_found[0].attrs['href'], "/title/", "")[:-1]
    
    url_csv = 'https://drive.google.com/file/d/183653sts6cpCVv0anZjzJWIEhSOOsyip/view?usp=sharing'
    url_csv = 'https://drive.google.com/uc?id=' + url_csv.split('/')[-2]
    df = pd.read_csv(url_csv)
    
    ### The user choose a movie to compare. This movie is in the Top 1000?
    # If we didn't find the movie in the DataFrame -> Get movie data from imdb.com
    if df[ df['imdb_id'] == imdb_id_given_by_user ].shape[0] == 0:
        genres = []
        url = 'https://www.imdb.com/title/' + imdb_id_given_by_user
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')

        title = soup.find('h1').text
        plot = soup.find('span', attrs={'data-testid': 'plot-xl'}).text
        director = soup.select("a[href*='/name/nm']")[0].text
        year = soup.select("a[href*='/releaseinfo?ref_=tt_ov_rdat']")[0].text
        genres_soup = soup.select("a[href*='/search/title?genres=']")
        for genre in genres_soup:
            genres.append(genre.span.text)
        genres = ", ".join(genres)
        rate = float(soup.find('div', {'data-testid': 'hero-rating-bar__aggregate-rating__score'}).span.text)

        # Adding the new movie to the DataFrame
        new_movie = pd.DataFrame([{
                        'imdb_id': imdb_id_given_by_user,
                        'title': title,
                        'rate': rate,
                        'year': year,
                        'director': director,
                        'genres': genres,
                        'plot': plot }])

        df = pd.concat([df, new_movie], ignore_index=True)

    ### Concatenating 'genres' and 'director' with 'plot' to tokenize later
    # Here we dupplicate the director name so it has more weight comparing two movies
    df['combined_features'] = df['title'] + " " + df['director'] + " " + \
                        df['genres'].apply(lambda x: str.replace(x, '\'', '')) + \
                        " " + df['plot']
    

    # testing the function:
    # print( tokenize_and_stem("[Drama, Romance, War] At a U.S. Army base at 1945") )
    ### Vectorizing the movie plot and genres with TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer( max_features=50000, stop_words='english',
                                        tokenizer=tokenize_and_stem, ngram_range=(1,2) )

    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'].values)

    ### Calculating the Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    movie_index = df[ df['imdb_id'] == imdb_id_given_by_user ].index

    # Find the cosine similarity tax with all the movies from the df, addint the index in the tupple
    similar_movies = list(enumerate(cosine_sim[movie_index,:][0]))

    # Sort the result, in descending order
    sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)
    
    ### Finding the data from the 10 movies more simmilar to the user's choice
    top_15_indexes = [ m[0] for m in sorted_similar_movies[:15] ]
    top_15_scores = [ m[1] for m in sorted_similar_movies[:15] ]
    top_15_sim = df.iloc[top_15_indexes].drop(['combined_features'], axis=1)
    top_15_sim['similarity'] = top_15_scores
    
    return top_15_sim.to_string()