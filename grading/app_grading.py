import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import json
import numpy as np
from essayGradeHelper import essay_to_wordlist, get_model, getAvgFeatureVecs

app = Flask(__name__)
CORS(app)

# Configure Google Gemini
gemini_key = ''
genai.configure(api_key=gemini_key)
model_name = "gemini-pro-vision"
ocr = genai.GenerativeModel(model_name)

# Word2Vec and LSTM model paths
word2vec_path = 'word2vecmodel (1).bin'
lstm_model_path = 'final_lstm (1).h5'
num_features = 300

# Load SentenceTransformer for synoptic grading
synoptic_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Load Word2Vec model for essay grading
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Essay grading API
@app.route("/grade_essay", methods=["POST"])
def grade_essay():
    data = request.get_json()
    essay_content = data.get('essay_content')
    
    if not essay_content:
        return jsonify({"error": "Essay content is missing."}), 400
    
    if len(essay_content) > 4:
        clean_test_essays = [essay_to_wordlist(essay_content, remove_stopwords=True)]
        testDataVecs = getAvgFeatureVecs(clean_test_essays, word2vec_model, num_features)
        testDataVecs = np.array(testDataVecs).reshape((len(testDataVecs), 1, num_features))
        
        lstm_model = get_model()
        lstm_model.load_weights(lstm_model_path)
        preds = lstm_model.predict(testDataVecs)
        predicted_grade = float(preds[0][0])
        return jsonify({"predicted_grade": predicted_grade})

    return jsonify({"error": "Essay content too short."}), 400

# Synoptic grading API
@app.route("/grade_synoptic", methods=["POST"])
def grade_synoptic():
    data = request.get_json()
    question_text = data.get('question_text')
    synoptic_text = data.get('synoptic_text')
    student_answer = data.get('student_answer')

    if not (question_text and synoptic_text and student_answer):
        return jsonify({"error": "Incomplete input."}), 400

    # Calculate similarity score
    synoptic_vec = synoptic_model.encode([synoptic_text])[0]
    student_vec = synoptic_model.encode([student_answer])[0]

    similarity_score = 1 - distance.cosine(synoptic_vec, student_vec)
    len_norm_factor = min(len(synoptic_text), len(student_answer)) / max(len(synoptic_text), len(student_answer))
    similarity_score *= len_norm_factor

    return jsonify({
        "similarity_score": similarity_score,
        "feedback": "High similarity" if similarity_score > 0.8 else 
                    "Moderate similarity" if similarity_score > 0.5 else "Low similarity"
    })

if __name__ == "__main__":
    app.run(host="localhost", port=9002)
