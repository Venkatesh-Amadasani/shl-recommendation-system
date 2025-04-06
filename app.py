from flask import Flask, request, render_template, jsonify
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)
assessments = [
    {"name": "Verify G+", "url": "https://www.shl.com/solutions/products/verify-g-plus/", "description": "General ability test for cognitive skills", "remote": "Yes", "adaptive": "Yes", "duration": "30 mins", "test_type": "Cognitive"},
    {"name": "Coding Skills - Java", "url": "https://www.shl.com/solutions/products/coding-skills-java/", "description": "Assess Java programming proficiency", "remote": "Yes", "adaptive": "No", "duration": "45 mins", "test_type": "Technical"},
    {"name": "Work Behaviors", "url": "https://www.shl.com/solutions/products/work-behaviors/", "description": "Evaluate collaboration and teamwork", "remote": "Yes", "adaptive": "No", "duration": "20 mins", "test_type": "Personality"},
    {"name": "Python Skills", "url": "https://www.shl.com/solutions/products/python-skills/", "description": "Test Python coding ability", "remote": "Yes", "adaptive": "Yes", "duration": "40 mins", "test_type": "Technical"},
    {"name": "SQL Proficiency", "url": "https://www.shl.com/solutions/products/sql-proficiency/", "description": "Measure SQL database skills", "remote": "Yes", "adaptive": "No", "duration": "35 mins", "test_type": "Technical"},
    {"name": "Cognitive Ability", "url": "https://www.shl.com/solutions/products/cognitive-ability/", "description": "Broad cognitive assessment", "remote": "Yes", "adaptive": "Yes", "duration": "25 mins", "test_type": "Cognitive"},
    {"name": "Personality Questionnaire", "url": "https://www.shl.com/solutions/products/personality-questionnaire/", "description": "Assess personality traits", "remote": "Yes", "adaptive": "No", "duration": "30 mins", "test_type": "Personality"},
    {"name": "JavaScript Skills", "url": "https://www.shl.com/solutions/products/javascript-skills/", "description": "Evaluate JavaScript coding", "remote": "Yes", "adaptive": "No", "duration": "40 mins", "test_type": "Technical"},
    {"name": "Numerical Reasoning", "url": "https://www.shl.com/solutions/products/numerical-reasoning/", "description": "Test numerical aptitude", "remote": "Yes", "adaptive": "Yes", "duration": "20 mins", "test_type": "Cognitive"},
    {"name": "Team Dynamics", "url": "https://www.shl.com/solutions/products/team-dynamics/", "description": "Measure team collaboration skills", "remote": "Yes", "adaptive": "No", "duration": "25 mins", "test_type": "Personality"}
]
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract meaningful text (e.g., paragraphs, headings)
        text = ' '.join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3']))
        return text.strip()
    except Exception as e:
        return f"Error fetching URL: {str(e)}"
def extract_duration(text):
    match = re.search(r'(\d+)\s*(mins|minutes)', text.lower())
    return int(match.group(1)) if match else None
def recommend_assessments(input_text, max_duration=None):
    corpus = [f"{a['name']} {a['description']}" for a in assessments]
    vectorizer = TfidfVectorizer(stop_words='english')
    assessment_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([input_text])
    
    similarities = cosine_similarity(query_vector, assessment_vectors).flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    
    recommendations = []
    for idx in ranked_indices:
        assessment = assessments[idx]
        duration_mins = int(assessment['duration'].split()[0])
        if max_duration and duration_mins > max_duration:
            continue
        recommendations.append(assessment)
        if len(recommendations) >= 10:
            break
    return recommendations if recommendations else [assessments[ranked_indices[0]]]
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    error = None
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        url = request.form.get('url', '').strip()
        
        if url and query:
            error = "Please provide either a query or a URL, not both."
        elif url:
            input_text = extract_text_from_url(url)
            if input_text.startswith("Error"):
                error = input_text
            else:
                max_duration = extract_duration(input_text)
                recommendations = recommend_assessments(input_text, max_duration)
        elif query:
            max_duration = extract_duration(query)
            recommendations = recommend_assessments(query, max_duration)
        else:
            error = "Please provide a query or URL."
    
    return render_template('index.html', recommendations=recommendations, error=error)

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    query = request.args.get('query', '').strip()
    url = request.args.get('url', '').strip()
    
    if url and query:
        return jsonify({"error": "Please provide either a query or a URL, not both"}), 400
    elif url:
        input_text = extract_text_from_url(url)
        if input_text.startswith("Error"):
            return jsonify({"error": input_text}), 400
        max_duration = extract_duration(input_text)
        recommendations = recommend_assessments(input_text, max_duration)
    elif query:
        max_duration = extract_duration(query)
        recommendations = recommend_assessments(query, max_duration)
    else:
        return jsonify({"error": "Please provide a query or URL"}), 400
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)