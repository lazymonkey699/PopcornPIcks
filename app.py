from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import hashlib
import pandas as pd
from tmdbv3api import TMDb, Movie
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from contextlib import closing
from functools import wraps
import pickle

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Secret key for session management

# TMDb API setup
tmdb = TMDb()
tmdb.api_key = 'abb363d772ca1a1419a81d675eebc381'  # Replace with your TMDb API key
tmdb_movie = Movie()

# Lazy initialization of the dataset and similarity matrix
df = None  # DataFrame to hold movie data
similarity_matrix = None  # Matrix to hold similarity scores between movies

def load_data():
    """Load movie data and compute similarity matrix if not already loaded."""
    global df, similarity_matrix
    if df is None:
        # Load movie data into DataFrame
        df = pd.read_csv(r"D:\recommend\Data_sets\movies.csv")
        # Combine relevant features into a single string for each movie
        df["combined_features"] = df[['keywords', 'cast', 'genres', 'director']].fillna('').agg(' '.join, axis=1)
    
    if similarity_matrix is None:
        similarity_matrix = load_or_compute_similarity()

def load_or_compute_similarity():
    """Load the similarity matrix from disk, or compute and save it if not available."""
    try:
        with open('similarity_matrix.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, IOError):
        # Compute similarity matrix if not found
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df['combined_features'])
        similarity = cosine_similarity(count_matrix)
        # Save the computed matrix to disk
        with open('similarity_matrix.pkl', 'wb') as f:
            pickle.dump(similarity, f)
        return similarity

# Cache to store movie data fetched from TMDb API
movie_data_cache = {}

def get_movie_data(title):
    """Fetch movie poster, rating, and description from TMDb API."""
    if title in movie_data_cache:
        return movie_data_cache[title]

    search = tmdb_movie.search(title)
    if search:
        movie = search[0]
        full_path = f"https://image.tmdb.org/t/p/w500{movie.poster_path}"
        rating = movie.vote_average
        description = movie.overview
        movie_data_cache[title] = (full_path, rating, description)
        return full_path, rating, description
    
    # If movie is not found
    movie_data_cache[title] = (None, None, "Description not available.")
    return None, None, "Description not available."

# Database connection management functions
def create_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect('popcornpicks.db')
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query, params=()):
    """Execute a query that modifies the database."""
    with create_connection() as conn, closing(conn.cursor()) as cursor:
        cursor.execute(query, params)
        conn.commit()

def fetch_query(query, params=()):
    """Fetch data from the database."""
    with create_connection() as conn, closing(conn.cursor()) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()

# Database initialization
def init_db():
    """Initialize the database with necessary tables."""
    execute_query('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    
    execute_query('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            poster TEXT,
            status TEXT DEFAULT 'unwatched',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

# Authentication decorator
def login_required(f):
    """Decorator to ensure a user is logged in before accessing certain routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# User functions
def add_user(username, password):
    """Add a new user to the database with a hashed password."""
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    execute_query("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))

def verify_user(username, password):
    """Verify if the username and password match an existing user."""
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return fetch_query("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))

def get_user_id(username):
    """Get the user ID for a given username."""
    result = fetch_query("SELECT id FROM users WHERE username = ?", (username,))
    return result[0]['id'] if result else None

# Watchlist functions
def add_to_watchlist(user_id, movie_title, poster):
    """Add a movie to the user's watchlist."""
    execute_query("INSERT INTO watchlist (user_id, movie_title, poster) VALUES (?, ?, ?)", (user_id, movie_title, poster))

def get_watchlist(user_id):
    """Retrieve the user's watchlist."""
    return fetch_query("SELECT movie_title, poster, status FROM watchlist WHERE user_id = ?", (user_id,))

def update_watchlist_status(user_id, movie_title, status):
    """Update the status of a movie in the user's watchlist."""
    execute_query("UPDATE watchlist SET status = ? WHERE user_id = ? AND movie_title = ?", (status, user_id, movie_title))

# Movie recommendation function
def recommend_movies(movie_title):
    """Recommend movies similar to the given movie title."""
    load_data()
    find_close_match = difflib.get_close_matches(movie_title, df["title"].tolist())
    if not find_close_match:
        return []
    
    index_of_the_movie = df[df.title == find_close_match[0]].index[0]
    similarity_score = list(enumerate(similarity_matrix[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score[1:], key=lambda x: x[1], reverse=True)[:20]
    return [df.iloc[movie[0]]["title"] for movie in sorted_similar_movies]

# Flask routes
@app.before_request
def initialize():
    """Initialize the database before handling requests."""
    if not getattr(app, 'db_initialized', False):
        init_db()
        app.db_initialized = True

@app.route('/')
def home():
    """Render the home page (login page)."""
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = verify_user(username, password)
        if user:
            session['authenticated'] = True
            session['username'] = username
            return redirect(url_for('recommendation'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user signup."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        add_user(username, password)
        flash('Signup successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/recommendation', methods=['GET', 'POST'])
@login_required
def recommendation():
    """Render movie recommendations based on user input."""
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommend_movies(movie_name)
        
        movies_with_details = [
            {
                'title': movie,
                'poster': poster,
                'rating': rating,
                'description': description
            } for movie in recommendations 
            if (poster := get_movie_data(movie)[0])
            if (rating := get_movie_data(movie)[1])
            if (description := get_movie_data(movie)[2])
        ]
        return render_template('recommendations.html', movies_with_details=movies_with_details)
    return render_template('recommendations.html', movies_with_details=[])

@app.route('/logout', methods=['POST'])
def logout():
    """Log the user out and clear the session."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/add_to_watchlist', methods=['POST'])
def add_to_watchlist_route():
    """API endpoint to add a movie to the user's watchlist."""
    if not session.get('authenticated'):
        return jsonify(success=False, message="User not authenticated")

    user_id = get_user_id(session['username'])
    data = request.json
    title = data.get('title')
    poster = data.get('poster')

    if user_id and title:
        add_to_watchlist(user_id, title, poster)
        return jsonify(success=True)
    return jsonify(success=False, message="Failed to add to watchlist")

@app.route('/watchlist')
@login_required
def watchlist():
    """Render the user's watchlist."""
    user_id = get_user_id(session['username'])
    watchlist = get_watchlist(user_id)
    return render_template('watchlist.html', watchlist=watchlist)

@app.route('/update_status', methods=['POST'])
def update_status():
    """API endpoint to update the watchlist status of a movie."""
    user_id = get_user_id(session['username'])
    data = request.json
    title = data.get('title')
    status = data.get('status')
    
    if user_id and title and status:
        update_watchlist_status(user_id, title, status)
        return jsonify(success=True)
    return jsonify(success=False)

if __name__ == "__main__":
    app.run(debug=True)
