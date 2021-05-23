import os

from flask import Flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

def create_app(test_config=None):
    app = Flask(__name__)

    # a simple page that says hello
    @app.route('/')
    def hello():
        return 'Hello, World!'

    @app.route('/text-similarity')
    def text_similarity():
        sentences = [
            'The cat is cute',
            'All my cats in a row',
            'When my cat sits down, she looks like a Furby toy',
            'The cat from outer space',
            'Sunshine lovers to sit like this for some reason.'
            'The dog is laying around',
            'The cat is nice'
        ]
        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(sentences).todense()
        print(vectorizer.vocabulary_)

        for feature in features:
            print(euclidean_distances(features[0], feature))

        return 'Here is the similar items!'

    return app
