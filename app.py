from flask import Flask, request, jsonify
from flasgger import Swagger

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

print("Tensorflow:", tf.__version__)
print("Tensorflow Hub:", hub.__version__)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
app = Flask(__name__)
swagger = Swagger(app)

@app.route('/use/embed', methods=['POST'])
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
  from waitress import serve
  serve(app, host="0.0.0.0", port=5050)
  print("server running")