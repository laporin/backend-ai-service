import os

import requests
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

        response = requests.get('https://laporin.arifikhsanudin.my.id/api/reports/all')
        reports = response.json()['data']
        sentences = [report['detail'] for report in reports]

        sentences.insert(0, sentence)

        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(sentences).todense()

        similar_reports = []

        for index in range(len(features)):
            feature = features[index]
            sentence = sentences[index]
            distance = euclidean_distances(features[0], feature)[0][0]
            print(f'{index} - {distance} - {sentence}')

            for report in reports:
                if (report['detail'] == sentence):
                    report['distance'] = distance

        reports.sort(key=lambda report: report['distance'])  # sort by distance
        reports = reports[:3]  # take first third element

        print('---------------')
        print(reports)
        print([report['distance'] for report in reports])

        return jsonify({'data': reports})

    @app.route('/api/cat-similarity', methods=['POST'])
    def cat_similarity():
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
