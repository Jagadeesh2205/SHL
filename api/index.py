"""
Vercel-optimized Flask API for SHL Assessment Recommendation System
Ultra-minimal version - NO external imports from project
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Cache for assessments
_assessments_cache = None


def get_data_path():
    """Get the absolute path to data file"""
    # Try multiple possible locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'scraped_data.json'),
        os.path.join(os.getcwd(), 'data', 'scraped_data.json'),
        '/var/task/data/scraped_data.json',  # Vercel Lambda path
        'data/scraped_data.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return possible_paths[0]  # Default to first option


def load_assessments():
    """Load assessments with error handling"""
    global _assessments_cache
    
    if _assessments_cache is not None:
        return _assessments_cache
    
    try:
        data_path = get_data_path()
        with open(data_path, 'r', encoding='utf-8') as f:
            # Read in chunks to avoid memory issues
            content = f.read()
            _assessments_cache = json.loads(content)
        return _assessments_cache
    except Exception as e:
        # Return empty list if can't load
        return []


def simple_recommend(query, k=10):
    """Ultra-simple keyword matching"""
    assessments = load_assessments()
    
    if not assessments:
        return []
    
    query_lower = query.lower()
    words = [w for w in query_lower.split() if len(w) > 3]
    
    # Score each assessment
    scored = []
    for assessment in assessments:
        score = 0
        text = f"{assessment.get('assessment_name', '')} {assessment.get('description', '')} {assessment.get('category', '')}".lower()
        
        # Simple word matching
        for word in words:
            if word in text:
                score += text.count(word)
        
        if score > 0:
            scored.append((assessment, score))
    
    # Sort and return top k
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored[:k]]


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy'}), 200


# Versioned health endpoint as per spec
@app.route('/v1/shl-assess/health', methods=['GET'])
def v1_health():
    return health()


@app.route('/recommend', methods=['POST'])
def recommend():
    """Get recommendations"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request'}), 400
        
        query = data.get('query', '').strip()
        if not query or len(query) < 3:
            return jsonify({'error': 'Invalid query'}), 400
        
        # Limit query length
        query = query[:1000]
        k = min(10, max(5, data.get('k', 10)))
        
        # Get recommendations
        recommendations = simple_recommend(query, k)
        
        # Format response
        result = []
        for rec in recommendations:
            result.append({
                'url': rec.get('url', ''),
                'name': rec.get('assessment_name', ''),
                'adaptive_support': rec.get('adaptive_support', 'No'),
                'description': (rec.get('description', '') or '')[:150],
                'duration': rec.get('duration', 0),
                'remote_support': rec.get('remote_support', 'Yes'),
                'test_type': [rec.get('test_type_full', rec.get('category', 'Other'))]
            })
        
        return jsonify({'recommended_assessments': result}), 200
    
    except Exception:
        return jsonify({'error': 'Internal error'}), 500


# Versioned recommend endpoint as per spec
@app.route('/v1/shl-assess/recommend', methods=['POST'])
def v1_recommend():
    return recommend()


@app.route('/api/info', methods=['GET'])
def info():
    """API info"""
    return jsonify({
        'name': 'SHL Assessment API',
        'version': '1.0.0',
        'status': 'operational'
    }), 200


# Vercel handler
app_handler = app


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting minimal Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)



if __name__ == '__main__':
    # For local testing
    port = int(os.getenv('PORT', 5000))
    print(f"Starting lightweight Flask API on port {port}")
    print("Using keyword-based recommendation (no ML models)")
    app.run(host='0.0.0.0', port=port, debug=True)
