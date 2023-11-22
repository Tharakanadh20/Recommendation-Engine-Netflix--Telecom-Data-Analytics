# app_with_recommendation.py

from flask import Flask, render_template, request
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the data, trained Decision Tree model, and TF-IDF vectorizer
with open(r'F:\VIT AP\Data Science & Analytics Internship\Netflix Movie Recommendation System\Netflix Movie Recommendation System\Netflix Movie Recommendation System\project_directory\data1.pkl', 'rb') as f:
    data = pickle.load(f)

with open(r'F:\VIT AP\Data Science & Analytics Internship\Netflix Movie Recommendation System\Netflix Movie Recommendation System\Netflix Movie Recommendation System\project_directory\best_decision_tree_model.pkl', 'rb') as f:
    best_dt_model = pickle.load(f)

with open(r'F:\VIT AP\Data Science & Analytics Internship\Netflix Movie Recommendation System\Netflix Movie Recommendation System\Netflix Movie Recommendation System\project_directory\tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


def recommend_movies_dt(movie_title, model, data, vectorizer, num_recommendations=5):
    # Find the index of the movie in the dataset
    idx = data[data['Title'] == movie_title].index[0]

    # Transform the movie description using the TF-IDF vectorizer
    movie_tfidf = vectorizer.transform([data.iloc[idx]['Tags__Summary']])

    # Predict the genre of the movie using the Decision Tree model
    predicted_genre = model.predict(movie_tfidf)[0]

    # Filter movies with the predicted genre
    recommended_movies = data[data['Genre'] == predicted_genre]

    # Exclude the input movie from the recommendations
    recommended_movies = recommended_movies[recommended_movies['Title'] != movie_title]

    # Get the top N recommended movie titles
    top_recommendations = recommended_movies.head(num_recommendations)['Title']

    return top_recommendations


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        if movie_title not in data['Title'].values:
            # Movie title not found
            error_message = f"Movie title '{movie_title}' not found in the dataset"
            return render_template('index.html', error_message=error_message)
        else:
            recommended_movies = recommend_movies_dt(
                movie_title, best_dt_model, data, tfidf_vectorizer)
            return render_template('recommendations.html', movie_title=movie_title, recommended_movies=recommended_movies)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
