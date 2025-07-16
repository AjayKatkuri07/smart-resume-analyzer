import os
from flask import Flask, render_template, request
from resume_parser import extract_text_from_pdf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)

    jobs = pd.read_csv('job_data.csv')
    jobs['combined'] = jobs['skills'] + ' ' + jobs['job_title']
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(jobs['combined'].tolist() + [resume_text])

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    match_index = similarity.argmax()
    match_score = round(similarity[0][match_index]*100, 2)

    matched_job = jobs.iloc[match_index]['job_title']
    required_skills = jobs.iloc[match_index]['skills']

    return render_template('result.html',
                           job=matched_job,
                           score=match_score,
                           skills=required_skills)

if __name__ == '__main__':
    app.run(debug=True)
    