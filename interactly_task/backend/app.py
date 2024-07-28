import pandas as pd
import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__) 

CORS(app)

@app.route('/')
def hello_world():
    data = {
        'message':'hello from flask!!'
    }
    return jsonify(data)




def preprocess_text(text):
    """
    Preprocesses text for better embedding generation (add more techniques as needed).
    """
    text = text.lower()
    # Add other preprocessing steps like removing stop words, stemming, etc.
    return text


# Load your dataset
df = pd.read_excel("RecruterPilot candidate sample input dataset.xlsx")

# Preprocess text
df['combined_text'] = df['Job Skills'] + ' ' + df['Experience'] + ' ' + df['Projects'] + ' ' + df['Comments']
df['combined_text'] = df['combined_text'].apply(preprocess_text)

# Create TF-IDF vectors (for initial filtering)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Choose a Sentence Transformer model
model_name = 'all-MiniLM-L6-v2'  # Replace with your desired model
sentence_model = SentenceTransformer(model_name)

# Generate embeddings for candidate data
embeddings = sentence_model.encode([df['combined_text'] for combined_text in df])

# Create FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

# Create a mapping between index and document ID
index_to_docstore_id = {i: doc for i, doc in enumerate(df['combined_text'].tolist())}


def find_matches(job_description):
    """
    Finds top matching candidates based on a given job description.

    Args:
        job_description (str): The job description text.

    Returns:
        pandas.DataFrame: A DataFrame containing top matching candidates with similarity scores.
    """

    # Preprocess job description
    jd_preprocessed = preprocess_text(job_description)

    # Retrieve similar candidates based on TF-IDF (for initial filtering)
    jd_vector = vectorizer.transform([jd_preprocessed])
    similarity_scores = cosine_similarity(jd_vector, tfidf_matrix).flatten()
    top_candidates = df.iloc[similarity_scores.argsort()[::-1][:10]]  # Get top 10 candidates

    # Create embeddings for top candidates and JD
    candidate_embeddings = sentence_model.encode(top_candidates['combined_text'].tolist())
    jd_embedding = sentence_model.encode([jd_preprocessed])[0]

    # Calculate similarity scores using embeddings
    similarity_scores = np.dot(candidate_embeddings, jd_embedding) / (np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(jd_embedding))
    top_candidates['similarity'] = similarity_scores

    # Rank candidates based on similarity scores
    top_candidates = top_candidates.sort_values(by='similarity', ascending=False)

     # Output formatted results
    print("Here are the shortlisted Top 5 Candidates list:")
    for index, candidate in top_candidates.iterrows():
        print(f"Name: {candidate['Name']}")
        print(f"Skills: {candidate['Job Skills']}")
        print(f"Experience: {candidate['Experience']}")
        print("\n")

    return top_candidates


# Example usage
# jd = "Data Scientist with 3+ years experience in Python"
# find_matches(jd)

@app.route('/getData', methods=['POST'])
def get_data():
    data = request.get_json()
    prompt = data.get('prompt', '')
    matches = find_matches(prompt)
    matchedDict = {}
    finalList = []
    for index, candidate in matches.iterrows():
        matchedDict['Name'] = candidate['Name']
        matchedDict['Job_Skills'] = candidate['Job Skills']
        matchedDict['Experience'] = candidate['Experience']
        finalList.append(matchedDict)
        matchedDict = {}
    print('//////////////matcheDict////////////')
    print(finalList)
    print('/////////////////////////////////')
    return jsonify({'dataframe': finalList})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)


