from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from dotenv import load_dotenv
from os import environ
from pathlib import Path
import json
import os
from extract import get_ents, extract_documents

# App Setup
app = Flask(__name__)
CORS(app)
api = Api(app)


class Extractor(Resource):
    def post(self):

        if request.is_json:
            paragraphs = request.json.get('paragraphs', [])
        else:
            items = list(request.form.items())
            paragraphs = [e[1] for e in items if e[0].startswith('paragraphs')]

        extractions = get_ents(paragraphs)

        return {
            'extractions': extractions
        }


api.add_resource(Extractor, '/extract')


class DocumentExtractor(Resource):
    def post(self):

        file_list = []

        # Get all filies in the request
        for e in request.files:
            
            file_list.append(request.files[e])
        
        print('file list', file_list)

        extractions = extract_documents(file_list)

        return {
            'extractions': extractions
        }

api.add_resource(DocumentExtractor, '/extractDocument')


if __name__ == '__main__':

    print('Starting up server ...')

    # Load configuration file
    config_path = os.path.join(os.path.realpath('.'), '.env')

    load_dotenv(dotenv_path=config_path)
    port = environ.get('PORT')

    app.run(debug=True, host='0.0.0.0', port=port)
