import itertools
import requests
from bs4 import BeautifulSoup
import nltk
import imdb
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import argparse
from nltk.corpus import stopwords
import gensim
from gensim.models import fasttext
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import aiohttp
from lxml import html
import sys
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
)

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTableWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTableWidget, QHBoxLayout, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import requests
import threading
import multiprocessing
from jinja2 import Template
import webbrowser

import openai

# Set your OpenAI API key
openai.api_key = "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"


def query_openai(prompt, model, max_tokens=1024):
    completion = openai.Completion.create(engine=model, prompt=prompt, max_tokens=max_tokens)
    return completion.choices[0].text

def write_html_table(html_table,title,year):

    template = """
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

        .movie {
        /* Add a shadow to the element */
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);

        /* Other styles for the element */
        width: 250px;
        height: auto;
        margin: 10px;
        background-color: #333333;
        overflow: hidden;
        cursor: pointer;
        border-radius: 5px; 
        }

        /* Change the appearance of the element on hover */
        .movie:hover {
        box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2), 0 12px 40px 0 rgba(0, 0, 0, 0.19);
        transform: scale(1.1);
        }


        .movie img {
        width: 100%;
        height: 60%;
        object-fit: cover;
        }

        .movie .info {
        padding: 10px;
        }

        .movie .info h3 {
        margin: 0;
        font-size: 20px;
        word-wrap: break-word;  
        overflow: hidden;
        text-overflow: ellipsis;
        }

        .movie .info p {
        margin: 10px 0;
        font-size: 14px;
        word-wrap: break-word;
        text-overflow: ellipsis;
        }

        .movie .info .rating {
        display: flex;
        align-items: center;
        font-size: 14px;
        color: #00b300;
        }

        .movie .info .rating i {
        margin-right: 5px;
        }
    </style>
    </head>
    <body>
    <h1>Similar Movies for {{ title }} ({{ year }})</h1>
    <div class="movie-container">
        {% for row in html_table %}
        <div class="movie">
            <a href="{{ row.url }}"><img src="{{ row.cover_url }}"></a>
            <div class="info">
            <h3>{{ row.title }} ({{ row.year }})</h3>
            <p>{{ row.plot }}</p>
            <p><strong>Director: {{ row.director }}</strong></p>
            <div class="rating">
                <i class="fa fa-star"></i>
                {{ row.rating }}
            </div>
            </div>
        </div>
        {% endfor %}
    </div>
    </body>
    </html>
            """
    template = Template(template)
    html = template.render(html_table=html_table,title=title,year=year)

    with open(f'{title}_{year}.html', 'w') as f:
        f.write(html)

    webbrowser.open(f'{title}_{year}.html')


def load_model(model_path, queue):
    model = gensim.models.fasttext.load_facebook_vectors(model_path)
    queue.put(model)


def update_table(table, similar_movies):
    # Set the number of rows in the table
    table.setRowCount(len(similar_movies))

    # Set the values for each cell in the table
    for i in range(len(similar_movies)):
        # Set the values for each column in the row
        table.setItem(i, 0, QTableWidgetItem(similar_movies.iloc[i]["title"]))
        table.setItem(i, 1, QTableWidgetItem(similar_movies.iloc[i]["year"]))
        table.setItem(i, 2, QTableWidgetItem(similar_movies.iloc[i]["plot"]))
        #table.setItem(i, 3, QTableWidgetItem(similar_movies.iloc[i]["plot"]))
        table.setItem(i, 3, QTableWidgetItem(similar_movies.iloc[i]["director"]))
        table.setItem(i, 4, QTableWidgetItem(str(similar_movies.iloc[i]["rating"])))
        table.setItem(i, 5, QTableWidgetItem(similar_movies.iloc[i]["url"]))
        


def start_web_app():

    html="""
    <!DOCTYPE html>
<html>
<head>
    <title>Movie Recommendation</title>
</head>
<body>
    <form action="/" method="post">
        <label for="name">Movie name:</label><br>
        <input type="text" id="name" name="name"><br>
        <label for="year">Movie year:</label><br>
        <input type="text" id="year" name="year"><br>
        <label for="themes">Maximum number of themes:</label><br>
        <input type="text" id="themes" name="themes"><br>
        <label for="min_year">Min year:</label><br>
        <input type="text" id="min_year" name="min_year"><br>
        <label for="max_year">Max year:</label><br>
        <input type="text" id="max_year" name="max_year"><br>
        <label for="keywords">Keywords:</label><br>
        <input type="text" id="keywords" name="keywords"><br><br>
        <input type="submit" value="Search">
    </form> 
    <table style="width:100%">
        <tr>
            <th>Title</th>
            <th>Year</th> 
            <th>Plot</th>
            <th>Director</th>
            <th>Rating</th>
            <th>URL</th>
        </tr>
        {% for movie in movies %}
        <tr>
            <td>{{ movie.title }}</td>
            <td>{{ movie.year }}</td>
            <td>{{ movie.plot }}</td>
            <td>{{ movie.director }}</td>
            <td>{{ movie.rating }}</td>
            <td>{{ movie.url }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
    """



def start_gui(ia,queue):
    # Create the QApplication
    app = QApplication(sys.argv)

    # Create the main window
    window = QWidget()
    window.setWindowTitle("Movie Recommendation")

    # Create a label and a line edit for the movie name
    name_label = QLabel("Movie name:")
    name_edit = QLineEdit()

    # Create a label and a line edit for the movie year
    year_label = QLabel("Movie year:")
    year_edit = QLineEdit()

    # Create a label and a line edit for the maximum number of themes
    themes_label = QLabel("Maximum number of themes:")
    themes_edit = QLineEdit()

    # Create a label and a line edit for the min  years
    min_year_label = QLabel("Min year:")
    min_year_edit = QLineEdit()

    # Create a label and a line edit for the max  years
    max_year_label = QLabel("Max year:")
    max_year_edit = QLineEdit()

    # Create a label and a line edit for keywords to search for
    keywords_label = QLabel("Keywords:")
    keywords_edit = QLineEdit()


    # Create a label and a line edit for the movie year
    # plot_label = QLabel("Plot:")
    # plot_edit = QLineEdit()

    # Create a button to search for recommendations
    search_button = QPushButton("Search")
    
    # Create the table to display the recommendations
    table = QTableWidget()
    table.setColumnCount(6)
    table.setHorizontalHeaderLabels(["Title", "Year", "Plot", "Director", "Rating", "URL"])

    # Create a layout for the movie name and year inputs
    input_layout = QHBoxLayout()
    input_layout.addWidget(name_label)
    input_layout.addWidget(name_edit)
    input_layout.addWidget(year_label)
    input_layout.addWidget(year_edit)
    input_layout.addWidget(min_year_label)
    input_layout.addWidget(min_year_edit)
    input_layout.addWidget(max_year_label)
    input_layout.addWidget(max_year_edit)
    input_layout.addWidget(keywords_label)
    input_layout.addWidget(keywords_edit)
    # Add the label and line edit for the maximum number of themes to the input layout
    input_layout.addWidget(themes_label)
    input_layout.addWidget(themes_edit)
    # input_layout.addWidget(plot_label)
    # input_layout.addWidget(plot_edit)

    # Create the main layout
    main_layout = QVBoxLayout()
    main_layout.addLayout(input_layout)
    main_layout.addWidget(search_button)
    main_layout.addWidget(table)

    # Set the layout for the main window
    window.setLayout(main_layout)
    
    # Create a label to display the waiting message
    waiting_label = QLabel("Please wait while the search is being performed...")
    waiting_label.setAlignment(Qt.AlignCenter)
    waiting_label.hide()  # Hide the waiting label initially

    # Create a progress bar to show the progress of the search
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 0)  # Set the range to 0 to show an indeterminate progress bar
    progress_bar.hide()  # Hide the progress bar initially

    # Add the waiting label and progress bar to the main layout
    main_layout.addWidget(waiting_label)
    main_layout.addWidget(progress_bar)


    # Set up an HTTP session to make requests to the IMDb website
    s = requests.Session()

    # Set the user agent of the HTTP requests to mimic a web browser
    s.headers['User-Agent'] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:93.0) Gecko/20100101 Firefox/93.0'

    # Connect the search button to the event handler
    #search_button.clicked.connect(lambda: search_button_clicked(s, name_edit, year_edit, table, waiting_label, progress_bar, ia))
    
    search_button.clicked.connect(lambda: create_html_similar_movies(s,
                                                                    name_edit,
                                                                    year_edit,
                                                                    ia,
                                                                    table,
                                                                    queue,
                                                                    themes_edit,
                                                                    waiting_label,
                                                                    progress_bar,
                                                                    min_year_edit,
                                                                    max_year_edit,
                                                                    keywords_edit                                                                    
                                                                    ))

    # Show the main window
    window.show()

    # Run the application's event loop
    sys.exit(app.exec_())


def smooth_scale(x):
    # Apply a log transformation to x
    x_transformed = np.log(x + 1)
    lower_b=np.min(x_transformed)
    upper_b=np.max(x_transformed)
    if lower_b == upper_b:
        return x
    # Scale the transformed values to the range 0 to 1
    x_scaled = (x_transformed - np.min(x_transformed)) / (np.max(x_transformed) - np.min(x_transformed))
    
    return x_scaled

def tokenize_and_clean(words):
    # Use the nltk library to identify the part of speech of each word in the review body text
    words = nltk.word_tokenize(words)

    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiments = {word: sentiment_analyzer.polarity_scores(word) for word in words}

    # Filter the words to get only those that have a positive or negative sentiment
    filtered_sentiment_words = [word for word, scores in sentiments.items() if scores['neg'] > 0 or scores['pos'] > 0]

    # Include only adjectives and nouns in the list of filtered words
    pos_tags = nltk.pos_tag(words)
    filtered_words = [word[0] for word in pos_tags if word[1] in ['NN','JJ', 'JJR', 'JJS']]

    # Use the nltk library's stopwords data file to remove common words that are not likely to be related to the themes of the movie
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_sentiment_words = [word for word in filtered_sentiment_words if word not in stop_words]
    filtered_words = [word for word in filtered_words if word not in stop_words]

    # Use the nltk library's words data file to remove words that are not found in the dictionary
    words = nltk.corpus.words.words()
    clean_words = [word for word in filtered_words if word in words]
    
    # Remove the words found in filtered_sentiment_words from clean_words
    #clean_words = [word for word in clean_words if word not in filtered_sentiment_words]

    return clean_words


def get_liked_movies(s):

    import pandas as pd

    # Load the CSV file into a DataFrame
    df = pd.read_csv('watchlist.csv')
    movies = df[["Title","Year"]]
    for title, year in movies.to_records(index=False):
        create_html_similar_movies(s,title,year)


# Function to parse the HTML of a review page using lxml
def parse_page(html_text):
    tree = html.fromstring(html_text)
    review_texts = tree.xpath('//div[@class="content"]/div[@class="text show-more__control"]/text()')
    return review_texts

# Function to send an HTTP request using aiohttp and return the review text from the response
async def fetch_page(session, url):
    async with session.get(url) as response:
        html_text = await response.text()
        review_texts = parse_page(html_text)
        return review_texts

# Asynchronously send HTTP requests to the IMDb API to retrieve all of the review pages
async def get_review_pages(movieID, main_genres=[]):
    # Initialize an empty list to store the review pages
    review_pages = []

    # Initialize the aiohttp session
    async with aiohttp.ClientSession() as session:
        # Send the first HTTP request to the IMDb API to retrieve the first page of reviews
        url = f'https://imdb.com/title/{movieID}/reviews'
        review_page = await fetch_page(session, url)

    return review_page


def get_user_reviews_themes(model,movieID,main_genres=[]):
    # Initialize an empty list to store the filtered words
    all_filtered_words = []
    review_texts = asyncio.run(get_review_pages(f'tt{movieID}'))
    # Extract the review body text from the first page
    #review_texts = soup.find_all('div', {'class': 'content'})
    for review_text in review_texts:

        words = tokenize_and_clean(review_text)
        # Add the filtered words to the list of all filtered words
        all_filtered_words.append(words)
   
    all_filtered_words = list(set(itertools.chain(*all_filtered_words)))
    embedded_words = None#model.get_vector(" ".join(all_filtered_words))
    
    tagged_tokens = nltk.pos_tag(all_filtered_words)
    nouns = [word for word, pos in tagged_tokens if pos == "NN"]
    fd = nltk.FreqDist(nouns)
    common_nouns = [word for word, freq in fd.most_common(40)]
    main_genres_small = [word.lower() for word in main_genres]
    themes = list(set(main_genres_small+common_nouns))

    # Define a set of stopwords to filter out
    stop_words = set(stopwords.words('english'))

    # Filter the themes list to include only meaningful words, and not words like 'end'
    themes = [word for word in themes if word not in stop_words]

    # # Get the top k most common filtered words
    # #top_k_themes = [word[0] for word in nltk.FreqDist(all_filtered_words).most_common(num_themes)]
    # #return top_k_themes
    return themes, embedded_words

def filter_movies(queue,results,main_title,themes,embedded_themes,main_genres,similarity_metric="jaccard",min_year=None, max_year=None):

    themes_set = set(themes)
    movies=[]
    max_found = 10
    count=0
    use_user_reviews = True

    if similarity_metric == "cosine" or use_user_reviews:
        # Load the FastText model
        model = None #gensim.models.FastText.load_fasttext_format(r"C:\Users\sean\projects\moviescraper\crawl-300d-2M-subword\crawl-300d-2M-subword.bin")
        themes1_array = np.array(set(themes)) #np.array([model.wv[word] for word in themes_set]) #

    for i, movie in enumerate(results):
        print(i)
        try:
            
            # Search for the movie by title and year
            title = movie["title"]
            year = movie["year"]

            if min_year is not None and year < min_year:
                continue
            if max_year is not None and year > max_year:
                continue

            imdbid = movie.movieID
            movie_tmp = movie#ia.get_movie(str(imdbid)) #This object contains all the info
            
            if movie_tmp.get('rating',None) is None:
                rating = 0
            else:
                rating = movie_tmp['rating']

            plot=movie_tmp['plot'][0]
            director=movie_tmp['director'][0]['name']
            kind = movie['kind']
            genre=','.join(movie_tmp['genres'])

            if movie_tmp.get('top 250 rank',None) is None:
                top_250_rank = 0
            else:    
                top_250_rank = movie_tmp['top 250 rank']
            
            if movie_tmp.get('votes',None) is None:
                votes = 0
            else:    
                votes = movie_tmp['votes']            
            
            
            if title == main_title:
                raise TypeError("Same movie")
            if kind != 'movie':
                raise TypeError("Not a movie")
            if rating < 5.5:
                raise TypeError("Not a watchable movie")
            # if main_genres:
            #     set_1=set(movie_tmp['genres'])
            #     set_2=set(main_genres)

            #     #common_themes=set_1.intersection(set_2)
            #     #if not common_themes:
            #     if not (set_1.issubset(set_2) or set_2.issubset(set_1)):
            #         raise TypeError("No themes in common")

            cover=movie_tmp['full-size cover url']

            #ia.update(movie_tmp,['reviews'])
 
            if use_user_reviews:
                themes, embedded_themes = get_user_reviews_themes(model,imdbid,main_genres)
            else:
                themes = []

            plot_tokens  = tokenize_and_clean(plot)
  

            set2 = set(themes)

            if similarity_metric == "jaccard":
                similarity = len(themes_set.intersection(set2)) / len(themes_set.union(set2)) + np.random.rand()*(10**-3) 
            else:
                
                # Convert the words to their FastText embeddings
                themes2_array = np.array([model.wv[word] for word in set2])

                # Compute the dot product of the vectors
                similarity_matrix = cosine_similarity(themes1_array, themes2_array)

                max_cosine_similarities = np.max(similarity_matrix, axis=1)

                # Take the average of the maximum cosine similarities
                similarity = np.mean(max_cosine_similarities)   

            # Define a dictionary containing a movie's fields
            print("Title:", title)
            print("Year:", year)
            print("Similarity:", similarity)
            movie_d = {'title': title, 'year': year, 'rating': rating, 'similarity': similarity, 'genre':genre, 'plot': plot, 'director':director, 'cover_url': cover, 'votes': votes, 'top 250 rank': top_250_rank, 'url': f'https://www.imdb.com/title/tt{imdbid}'}
            if len(movies) == 0:
                heapq.heappush(movies, (movie_d['similarity'], movie_d))
                heapq.heapify(movies)
                count+=1
                
            # If the heap is full and the current movie has a higher similarity than the movie with the lowest similarity, pop the movie with the lowest similarity
            elif len(movies) == max_found and similarity > movies[0][0]:
                heapq.heappop(movies)
                heapq.heappush(movies, (movie_d['similarity'], movie_d))     
           
            elif len(movies) < max_found :
                # Push the current movie and its similarity to the heap
                heapq.heappush(movies, (movie_d['similarity'], movie_d))     
                count+=1
            
            print(f"Count: {count}")

        except:
            continue

    # Extract the dictionaries from the heap and add them to a list
    movies_list = []
    while len(movies) > 0:
        movies_list.append(heapq.heappop(movies)[1])    

    return pd.DataFrame(movies_list)

# Define a function to update a movie with the rating information
def update_movies(movies,queue):
    for movie in movies:
        try:
            ia.update(movie)
        except:
            continue
    # Sort the movies by rating in descending order
    #sorted_movies = sorted(movies, key=lambda movie: movie.get('rating', 0), reverse=True)

    # Take the top 10 movies
    #max_r = min(len(sorted_movies),40)
    #results = sorted_movies[:max_r]
    queue.put(movies)



def find_similar_movies(soup, title, movieID, num_themes=500,main_genres=[],queue=None, plot_tokens=None,ia=None, table=None, min_year=None, max_year=None,keywords=None):
    
    themes, embedded_themes = get_user_reviews_themes(queue, movieID,main_genres)
    themes = keywords + plot_tokens + themes
    

    
    model = "text-davinci-002"
    prompt = f"Suggest me your top 10 movies like {title}. The following are themes by users reviews that should be respected: {themes[:num_themes]}."

    # query the openai api
    response = query_openai(prompt,model)
    # Use a list comprehension to get the keywords for each theme and store the results in a list
    #results = [ia.get_keyword(word) for word in themes[:10]]
    results=[]
    # Create a queue
    output_queue = multiprocessing.Queue()
    # Create a list of threads
    threads = []
    for word in themes[:num_themes]:
        movies = ia.get_keyword(word)
        if len(movies) == 0:
            continue

        # Create a thread to update the movie
        thread = threading.Thread(target=update_movies, args=(movies,output_queue))

        # Add the thread to the list
        threads.append(thread)

        # Start the thread
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Retrieve the results from the queue
    while not output_queue.empty():
        results.append(output_queue.get())

    
    results=list(set(itertools.chain(*results)))
    results = sorted( results, key=lambda movie: movie.get('rating', 0), reverse=True)
    results_subset=results[:50]
    print(f"Found {len(results_subset)} movies.")
    similar_movies = filter_movies(queue,results_subset,title,themes,embedded_themes,main_genres,min_year, max_year)

    if similar_movies.empty:
        print("Nothing was found.")
        return None
    else:
        similar_movies["rating_scaled"] = pd.Series(smooth_scale(similar_movies["rating"].values))
        similar_movies["similarity_scaled"] = pd.Series(smooth_scale(similar_movies["similarity"].values))
        similar_movies["votes"] = pd.Series(smooth_scale(similar_movies["votes"].values))
        similar_movies['top 250 rank'] = pd.Series(smooth_scale(similar_movies['top 250 rank'].values))
        similar_movies['score'] = (similar_movies['similarity_scaled'] + similar_movies['rating_scaled'] + 0.5*similar_movies['top 250 rank'] + 0.25*similar_movies['votes']).abs()
        return similar_movies



def create_html_similar_movies(s,title,year,ia,table,queue,themes_edit,waiting_label,progress_bar, min_year_edit, max_year_edit,keywords_edit):

    max_themes = int(themes_edit.text())

    if min_year_edit.text() == '':
        min_year = None
    else:
        min_year = int(min_year_edit.text())
    if max_year_edit.text() == '':
        max_year = None
    else:
        max_year = int(max_year_edit.text())
    if keywords_edit.text() == '':
        keywords_edit = ['']
    else:
        keywords_edit.text()
        keywords_edit = keywords_edit.split(',')

    search = ia.search_movie(title.text(), year.text().strip())
    movie = search[0]
    imdbid = movie.movieID
    movie_tmp = ia.get_movie(str(imdbid)) #This object contains all the info
    main_genres=movie_tmp['genres']
    main_plot = movie_tmp['plot'][0]

    plot_tokens  = tokenize_and_clean(main_plot)
    keywords_tokens = tokenize_and_clean(' '.join(keywords_edit))

    url = f'https://www.imdb.com/title/tt{imdbid}/reviews?ref_=tt_urv'
    response = s.get(url)

    # Parse the HTML response using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    title = movie_tmp['title']
    year = movie_tmp['year']
    print(f"Movie: {title} ({year})")

    #Copy code
    # Clear the table
    table.setRowCount(0)

    # Show the waiting label and progress bar
    waiting_label.show()
    progress_bar.show()

    similar_movies = find_similar_movies(soup, title, imdbid, max_themes, main_genres, queue,plot_tokens,ia, table, min_year, max_year,keywords_tokens)
    if similar_movies is None:
        return None

    similar_movies=similar_movies.sort_values(by=['score'],ascending=False)

    update_table(table, similar_movies[["title","year","plot", "director", "rating", "url"]])

    # Hide the waiting label and progress bar
    waiting_label.hide()
    progress_bar.hide()


    # import sqlite3

    # # Connect to the database
    # conn = sqlite3.connect('movies.db')

    # # Save the DataFrame to the database
    # #similar_movies.to_sql(f"{movie_tmp['title']}", conn, if_exists='replace')
    # # Save the DataFrame to the database
    # similar_movies.to_sql(f"{movie_tmp['title']}", conn, if_exists='append')

    # # Close the connection
    # conn.close()

    # Convert the DataFrame to an HTML table
    #html_table = similar_movies[["cover_url","title","year","plot", "director", "rating", "url"]].to_html(escape=False, formatters=dict(cover_url=path_to_image_html))
    print(similar_movies[["cover_url","title","year","plot", "genre","director", "rating", "url"]].head(10))
    #html_table = similar_movies[["cover_url","title","year","plot", "director", "rating", "url"]].to_dict()
    html_table  = [row.to_dict() for _, row in similar_movies.iterrows()]
    write_html_table(html_table,title,year)
    
    return similar_movies

# Converting links to html tags
def path_to_image_html(path):
    return '<img src="'+ path + '" width="250" >'

if __name__ == "__main__":

    queue=None
    use_gui=True
    # Create an instance of the IMDb class
    ia = imdb.Cinemagoer()

    # Download the nltk data files
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

    if use_gui:
        start_gui(ia,queue)
    else:
        # Set up an HTTP session to make requests to the IMDb website
        s = requests.Session()

        # Set the user agent of the HTTP requests to mimic a web browser
        s.headers['User-Agent'] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:93.0) Gecko/20100101 Firefox/93.0'
        #create_html_similar_movies(s,name_edit,year_edit,ia,table,queue,themes_edit,waiting_label,progress_bar)

