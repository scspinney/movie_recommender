import numpy as np
import pandas as pd
import re
import nltk
from imdb import IMDb
import multiprocessing as mp
from nltk.stem.snowball import SnowballStemmer


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from jinja2 import Template
#from webui import WebUI

from flask import Flask, request, render_template


app = Flask(__name__)
#ui = WebUI(app, debug=True) # Create a WebUI instance


@app.route('/', methods=['GET', 'POST'])
def index():
  movies_df = get_movie_df()
  with app.app_context():
    return start_web_app(movies_df)


@app.route('/select-movie', methods=['POST', 'GET'])
def select_movie():
  # Parse the request body
  data = request.get_json()
  
  # Access the selected movie's title, year, and url
  #id = data['id']
  title = data['title']
  movies_df=get_movie_df()
  tfidf = TfidfVectorizer(stop_words='english') 
  tfidf_matrix = tfidf.fit_transform([str(x) for x in movies_df['plot_combined']])
  #tfidf_matrix = pipe.fit_transform([str(x) for x in movies_df['plot']])

  similarity_distance = 1- cosine_similarity(tfidf_matrix)
  most_similar = find_similar(movies_df,similarity_distance,title,num_recommendations=5)
  most_similar['title'] = most_similar['Title']
  most_similar['year'] = most_similar['Year']
  most_similar['url'] = most_similar['URL']
  most_similar['directors'] = most_similar['Directors']
  most_similar['rating'] = most_similar['IMDb Rating']
  most_similar['genres'] = most_similar['Genres']
  #most_similar['similarity'] = similarity_distance
  html_table = [row.to_dict() for _, row in most_similar.iterrows()]
  #html_table = [{'title': title, 'year': year, } for title, year in zip(titles, years) if title != main_title] 
  #html_table = [{'title': title, 'year': year}] + html_table 
  template = """
    <!DOCTYPE html>
    <html>
    <head>  
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>
            strong {
            font-weight: bold;
            }

            /* Add some style to make the page look more like Netflix */
            body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            background-color: #262626;
            color: white;
            }

            h1 {
            margin: 20px;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            text-transform: uppercase;
            color: #00b8d4;
            }

            .movie-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin: 20px;
                /* Allow the content to be scrollable, but hide the scrollbar */
                overflow: auto;
                /* Hide the scrollbar in Internet Explorer and Edge */
                -ms-overflow-style: none;
                /* Hide the scrollbar in Chrome, Firefox, and Safari */
                scrollbar-width: none;
            }

            .movie-box {
            /* Add a shadow to the element */
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);

            /* Other styles for the element */
            width: 350px;
            height: 660px;
            margin: 10px;
            background-color: #333333;
            overflow: auto;
            /* Hide the scrollbar in Internet Explorer and Edge */
            -ms-overflow-style: none;
            /* Hide the scrollbar in Chrome, Firefox, and Safari */
            scrollbar-width: none;
            cursor: pointer;
            border-radius: 5px; 
            }

            /* Change the appearance of the element on hover */
            .movie-box:hover {
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2), 0 12px 40px 0 rgba(0, 0, 0, 0.19);
            transform: scale(1.1);
            }


            .movie-box img {
            width: 100%;
            height: 60%;
            object-fit: cover;
            }

            .movie-box .info {
            padding: 10px;
            }

            .movie-box .info h3 {
            margin: 0;
            font-size: 20px;
            word-wrap: break-word;  
            overflow: hidden;
            text-overflow: ellipsis;
            }

            .movie-box .info p {
            margin: 10px 0;
            font-size: 14px;
            word-wrap: break-word;
            text-overflow: ellipsis;
            white-space: normal;  /* Add the white-space property */
            }

            .movie-box .info .rating {
            display: flex;
            align-items: center;
            font-size: 14px;
            color: #00b300;
            }

            .movie-box .info .rating i {
            margin-right: 5px;
            }

            .btn.btn-secondary {
                /* Add the text-align property */
                text-align: center;
                /* Add the margin property */
                margin: 0 auto;
                /* Remove the underline */
                text-decoration: none;
                /* Add a floating box around the button */
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
            }
        </style>
    </head>
  <body>
    <h1>Recommendations for "{{ title }}"</h1>
    <div>
    <!-- Add the "Back" button with an onclick event handler -->
       <a href="/" onclick="history.back()" class="btn btn-secondary">Back</a>   
    </div>
       <div class="movie-container">
        {% for row in html_table %}
            <div class="movie-box">
                <a href="{{ row.url }}" class="movie" title="{{ row.title }}" year="{{ row.year }}" url="{{ row.url }}">
                <img src="{{ row.cover_url }}" alt="{{ row.title }}" class="cover">
                </a>
                <div class="info">
                <h3>{{ row.title }} ({{ row.year }})</h3>
                <p>{{ row.plot }}</p>
                <p><strong>Director: {{ row.directors }}</strong></p>
                <div class="rating">
                    <i class="fa fa-star"></i>
                    {{ row.rating }}
                </div>
                <div class="similarity">
                    <p><strong>Similarity: {{ row.similarity }}</strong></p>                    
                </div>
            </div>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
  template = Template(template)
  html = template.render(html_table=html_table,title=title)
    
    # Return the rendered template
  return html
  #return render_template(r'C:\Users\sean\projects\moviescraper\template.html', html_table=[{'title': title, 'year': year}])

def get_movie_df():
    import os

    # Check if the file "watchlist_complete.csv" exists
    if os.path.exists('watchlist_complete.csv'):
        # If the file exists, load it into a pandas DataFrame
        movies_df = pd.read_csv('watchlist_complete.csv')
    else:
        # If the file does not exist, load the "watchlist.csv" file
        movies_df = pd.read_csv('watchlist.csv')

    #movies_df=movies_df.iloc[:50, :]

    if movies_df['Description'].isnull().sum() == len(movies_df['Description']):
        #ia = IMDb()
        # Create a Pool with 4 worker processes
        with mp.Pool(4) as p:
        # Create an empty list to store the async results
            async_results = []
            
            # Loop through the movies_df['Const'] column
            for movie in movies_df['Const']:
                # Apply the function in parallel using the apply_async method
                
                async_result = p.apply_async(func=fetch_plot, args=(movie,))
                async_results.append(async_result)
                    # Use the starmap method to apply the function in parallel
                #results = p.starmap(fetch_plot, [(ia, movie) for movie in movies_df['Const']])
                
            # Use the get method of the AsyncResult objects to retrieve the results
            plots=[]
            cover_urls=[]
            for ar in async_results:
                result=ar.get()
                plots.append(result[0])
                cover_urls.append(result[1])

            #results = [ar.get() for ar in async_results]

        movies_df['plot'] = plots
        movies_df['plot'] = movies_df['plot'].astype(str)
        movies_df['cover_url'] = cover_urls
        movies_df['plot_combined'] = movies_df['plot'].astype(str) + "\n" + movies_df['Genres'].astype(str) + "\n" + movies_df['Directors'].astype(str)
        movies_df['genre'] = movies_df['Genres'].astype(str)
        movies_df['Description'] = movies_df['plot']
        movies_df.to_csv('watchlist_complete.csv', index=False)

    #start_web_app(movies_df)

    #tfidf = TfidfVectorizer(stop_words='english')
    #tfidf_matrix = tfidf.fit_transform([str(x) for x in movies_df['plot']])
    #tfidf_matrix = pipe.fit_transform([str(x) for x in movies_df['plot']])


    #similarity_distance = 1 - cosine_similarity(tfidf_matrix)
    return movies_df



def start_web_app(movies_df):
    
    template="""
       <!DOCTYPE html>
    <html>
    <head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    <style>
            strong {
            font-weight: bold;
            }

            /* Add some style to make the page look more like Netflix */
            body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            background-color: #262626;
            color: white;
            }

            h1 {
            margin: 20px;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            text-transform: uppercase;
            color: #00b8d4;
            }

            .movie-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            margin: 20px;
            }

            .movie-box {
          /* Add a shadow to the element */
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);

            /* Other styles for the element */
            width: 350px;
            height: 660px;
            margin: 10px;
            background-color: #333333;
            overflow: auto;
            /* Hide the scrollbar in Internet Explorer and Edge */
            -ms-overflow-style: none;
            /* Hide the scrollbar in Chrome, Firefox, and Safari */
            scrollbar-width: none;
            cursor: pointer;
            border-radius: 5px; 
            }

            /* Change the appearance of the element on hover */
            .movie-box:hover {
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2), 0 12px 40px 0 rgba(0, 0, 0, 0.19);
            transform: scale(1.1);
            }


            .movie-box img {
            width: 100%;
            height: 60%;
            object-fit: cover;
            }

            .movie-box .info {
            padding: 20px;
            }

            .movie-box .info h3 {
            margin: 0;
            font-size: 20px;
            word-wrap: break-word;  
            overflow: hidden;
            text-overflow: ellipsis;
            color: #ffffff; /* Use a light color for the text */
            }

            .movie-box .info p {
            margin: 10px 0;
            font-size: 14px;
            word-wrap: break-word;
            text-overflow: ellipsis;
            white-space: normal;  /* Add the white-space property */
            color: #b3b3b3; /* Use a lighter color for the text */
            }

            .movie-box .info .rating {
            display: flex;
            align-items: center;
            font-size: 14px;
            color: #00b300;
            font-size: 18px; /* Increase the font size */
            }

            .movie-box .info .rating i {
            margin-right: 5px;
            }

            .btn btn-primary {
            color: #03a9f4;
            /* Add the display and align-items properties */
            display: flex;
            align-items: center;
            }
        </style>
    </head>
    <body>
        <h1>Movies</h1>
<form method="POST" class="form-inline my-4">
  <div class="form-group mr-4">
    <label for="genre" class="mr-2 font-weight-bold">Genre:</label>
    <select id="genre" name="genre" class="form-control form-control-lg">
      <option value="all">All</option>
      <option value="Action">Action</option>
      <option value="Comedy">Comedy</option>
      <option value="Drama">Drama</option>
      <option value="Thriller">Thriller</option>
      <option value="Horror">Horror</option>
      <option value="Sci-Fi">Sci-Fi</option>
    </select>
  </div>
  <div class="form-group mr-4">
    <label for="rating" class="mr-2 font-weight-bold">Minimum Rating:</label>
    <select id="rating" name="rating" class="form-control form-control-lg">
      <option value="0">0</option>
      <option value="1">1</option>
      <option value="2">2</option>
      <option value="3">3</option>
      <option value="4">4</option>
      <option value="5">5</option>
      <option value="6">6</option>
      <option value="7">7</option>
      <option value="8">8</option>
      <option value="9">9</option>
      <option value="10">10</option>
    </select>
  </div>
  <div class="form-group mr-4">
    <label for="released" class="mr-2 font-weight-bold">Year Released:</label>
    <select id="released" name="released" class="form-control form-control-lg">
      <option value="all">All</option>
      <option value="2021">2021</option>
      <option value="2020">2020</option>
      <option value="2019">2019</option>
      <option value="2018">2018</option>
    </select>
  </div>
  <input type="submit" value="Filter" class="btn btn-primary btn-lg">
</form> 
        <div class="movie-container">
        {% for row in html_table %}
            <div class="movie-box">
                <a href="{{ row.url }}" class="movie" title="{{ row.title }}" year="{{ row.year }}" url="{{ row.url }}">
                <img src="{{ row.cover_url }}" alt="{{ row.title }}" class="cover">
                </a>
                <div class="info">
                <h2>{{ row.title }} ({{ row.year }})
                     <div class="rating">
                        <i class="fa fa-star"></i>
                        {{ row.rating }}
                    </div>
                </h2>
                <h3><strong>Genre: {{ row.genres }}</strong></h3>
                <p>{{ row.plot }}</p>
                <p><strong>Director: {{ row.directors }}</strong></p>
                <div class="similar-movies">
                    <a href="/select-movie" onclick="sendMovieData(event, '{{ row.title }}')" class="btn btn-primary" style="color:  #03a9f4;">Find Similar Movies</a>                
                </div>
            </div>
            </div>
            {% endfor %}
        </div>
    <!-- Add a script tag to include the sendMovieData function -->
    <script>
    function sendMovieData(event, title) {
    event.preventDefault(); // Prevent the default action of the anchor element
    // Send a POST request to the /select-movie route
    fetch('/select-movie', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({title: title})
    })
    .then(response => response.text())
    .then(html => {
        // Update the current page with the response from the /select-movie route
        document.body.innerHTML = html;
    });
    }
    </script>
    </body>
    </html>
    """

    # Render the template with the movie data
    movies_df['title'] = movies_df['Title']
    movies_df['year'] = movies_df['Year']
    movies_df['directors'] = movies_df['Directors']
    movies_df['rating'] = movies_df['IMDb Rating']
    movies_df['url'] = movies_df['URL']
    movies_df['genres'] = movies_df['Genres']
    movies_df_filtered = None
    
    genre = request.form.get('genre')  # retrieve the selected genre value from the form submission    
    if genre and genre != 'all':
        movies_df_filtered = movies_df[movies_df["genres"].str.contains(genre)]
        html_table = [row.to_dict() for _, row in movies_df_filtered.iterrows()]
    else:
        html_table = [row.to_dict() for _, row in movies_df.iterrows()]
    
    min_rating = request.form.get('rating') 
    if min_rating and min_rating != 'all':
        if movies_df_filtered is not None:
          movies_df_filtered = movies_df_filtered[movies_df_filtered["rating"] > int(min_rating)]
        else:
          movies_df_filtered = movies_df[movies_df["rating"] > int(min_rating)]
        html_table = [row.to_dict() for _, row in movies_df_filtered.iterrows()]
    else:
        html_table = [row.to_dict() for _, row in movies_df.iterrows()]

    template = Template(template)
    html = template.render(html_table=html_table)
       # Render the template with the movies data
    #html = render_template_string(template, movies_df=movies_df)
    
    # Return the rendered template
    return html


def fetch_plot(imdbID):
    ia = IMDb()
    imdbID = imdbID[2:]
    # search for a movie but use get_movie() to get the full info
    try:
        movie = ia.get_movie(imdbID)
        print(movie['title'])
        plot = movie['plot'][0]
        cover_url = movie['full-size cover url']
        return (plot, cover_url)
    except:
        return ('','')

    
def normalize(X): 
  stemmer = SnowballStemmer("english", ignore_stopwords=False)
  normalized = []
  for x in X:
    words = nltk.word_tokenize(x)
    normalized.append(' '.join([stemmer.stem(word) for word in words if re.match('[a-zA-Z]+', word)]))
  return normalized

def find_similar(movies_df,similarity_distance,title,num_recommendations=1):
  index = movies_df[movies_df['Title'] == title].index[0]
  vector = similarity_distance[index, :]
  movies_df['similarity'] = vector  
  most_similar = movies_df.iloc[np.argsort(vector)[:num_recommendations], [5,6,8,10,11,14,17,18,19,20,21]]

  return most_similar


def plot_dendogram(similarity_distance,movies_df):
    
    mergings = linkage(similarity_distance, method='complete')
    dendrogram_ = dendrogram(mergings,
                labels=[x for x in movies_df["Title"]],
                leaf_rotation=90,
                leaf_font_size=16,
                )

    fig = plt.gcf()
    _ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
    fig.set_size_inches(108, 21)

    plt.show()


if __name__ == '__main__':
    
    #print(find_similar('Seconds',5)) 
    app.debug = True
    #ui.run() #replace app.run() with ui.run(), and that's it
    app.run()