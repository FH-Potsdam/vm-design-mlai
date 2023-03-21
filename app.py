from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flasgger import Swagger
from waitress import serve

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

print("Tensorflow:", tf.__version__)
print("Tensorflow Hub:", hub.__version__)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
print("Model loaded")

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
swagger = Swagger(app)

@app.route('/use/embed', methods=['OPTIONS'])
@cross_origin()
def use_embed_post():
  return "", 200

@app.route('/use/embed', methods=['POST'])
@cross_origin()
def use_embed_post():
  """Returns embeddings from the universal-sentence-encoder-multilingual-large v3 for an array of strings
    ---
    parameters:
      - name: json
        in: body
        schema:
          type: object
          properties:
            text:
              type: array
              items:
                type: string
              required: true
              description: array of texts
    produces:
      - application/json
    accepts:
      - application/json
    responses:
      400:
        description: Input missing
      200:
        description: Array of embeddings vectors (512)
        examples:
          application/json: [[1,2,3,4,5],[1,2,3,4,5]]
  """
  if 'text' not in request.json or type(request.json['text']) is not list or len(request.json['text']) == 0:
    return 'Missing input', 400
  
  result = embed(request.json['text'])

  return jsonify(np.array(result).tolist()), 200


@app.route('/use/embed', methods=['GET'])
@cross_origin()
def use_embed_get():
  """Returns embeddings from the universal-sentence-encoder-multilingual-large v3 for a string
    ---
    parameters:
      - name: text
        in: query
        required: true
        schema:
          type: string
          description: text for embedding
    produces:
      - application/json
    responses:
      400:
        description: Input missing
      200:
        description: Embedding vectors (512)
        examples:
          application/json: [1,2,3,4,5]
  """
  text = request.args.get('text', default = None, type = str)
  if text is None:
    return 'Missing input', 400
  
  result = embed([text])

  return jsonify(np.array(result).tolist()[0]), 200

if __name__ == '__main__':
  print("server running on port 5050")
  serve(app, listen='*:5050')



