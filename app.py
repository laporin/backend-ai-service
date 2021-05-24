import os

from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


def create_app(test_config=None):
    app = Flask(__name__)

    # a simple page that says hello
    @app.route('/')
    def hello():
        return 'Hello world!'

    @app.route('/api/text-similarity', methods=['POST'])
    def text_similarity():
        sentence = request.json['detail']

        sentences = [
            'The cat is cute',
            'All my cats in a row',
            'When my cat sits down, she looks like a Furby toy',
            'The cat from outer space',
            'Sunshine lovers to sit like this for some reason.',
            'The dog is laying around',
            'The cat is nice'
        ]

        sentences.insert(0, sentence)

        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(sentences).todense()
        # print(vectorizer.vocabulary_)

        report = {'distance': 0, 'sentence': ''}

        for index in range(len(features)):
            feature = features[index]
            sentence = sentences[index]
            distance = euclidean_distances(features[0], feature)[0][0]
            print(f'{index} - {distance} - {sentence}')

            if (report['distance'] == 0):
                report['distance'] = distance
                report['sentence'] = sentence

            if (distance != 0 and distance < report['distance']):
                report['distance'] = distance
                report['sentence'] = sentence

        # return 'Here is the similar items!'
        return jsonify(report)

    return app
