"""
Flask API for SHL Assessment Recommendation System
Provides REST endpoints for health check and recommendations
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import RAGRecommender
from src.utils import validate_query, format_recommendations, is_url, fetch_jd_from_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Initialize recommender
logger.info("Initializing RAG Recommender...")
try:
    recommender = RAGRecommender()
    logger.info("Recommender initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    recommender = None


@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns API status
    """
    response = {
        'status': 'healthy'
    }
    
    return jsonify(response), 200


# Versioned health endpoint as per spec
@app.route('/v1/shl-assess/health', methods=['GET'])
def v1_health_check():
    return health_check()


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Assessment recommendation endpoint
    
    Request body:
    {
        "query": "I need a Java developer with good communication skills"
    }
    
    Response:
    {
        "query": "...",
        "recommendations": [
            {
                "assessment_name": "Java Programming Assessment",
                "url": "https://...",
                "relevance_score": 0.92,
                "test_type": "K",
                "category": "Knowledge & Skills",
                "description": "..."
            }
        ],
        "total_recommendations": 10
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        
        if 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query',
                'status': 'error'
            }), 400
        
        query = data['query']
        
        # Validate query
        if not validate_query(query):
            return jsonify({
                'error': 'Invalid query. Query must be non-empty and at least 3 characters.',
                'status': 'error'
            }), 400
        
        # Check if recommender is ready
        if recommender is None:
            return jsonify({
                'error': 'Recommender not initialized. Please run scraper and embeddings first.',
                'status': 'error'
            }), 503
        
        # Check if query is a URL
        if is_url(query):
            logger.info(f"Query is a URL, fetching content: {query}")
            jd_text = fetch_jd_from_url(query)
            
            if jd_text:
                query = jd_text[:2000]  # Use first 2000 chars
                logger.info(f"Extracted {len(query)} characters from URL")
            else:
                return jsonify({
                    'error': 'Failed to fetch content from URL',
                    'status': 'error'
                }), 400
        
        # Get number of recommendations (default 10, min 5, max 10)
        k = data.get('k', 10)
        k = max(5, min(10, k))
        
        # Generate recommendations
        logger.info(f"Generating {k} recommendations for query: {query[:100]}...")
        recommendations = recommender.recommend(query, k=k)
        
        # Format response
        response = {
            'recommended_assessments': format_recommendations(recommendations)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {e}", exc_info=True)
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500


# Versioned recommend endpoint as per spec
@app.route('/v1/shl-assess/recommend', methods=['POST'])
def v1_recommend():
    return recommend()


@app.route('/api/info', methods=['GET'])
def api_info():
    """
    API information endpoint
    Returns details about the API and available endpoints
    """
    info = {
        'name': 'SHL Assessment Recommendation API',
        'version': '1.0.0',
        'description': 'RAG-based recommendation system for SHL assessments',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /v1/shl-assess/health': 'Versioned health check',
            'POST /recommend': 'Get assessment recommendations',
            'POST /v1/shl-assess/recommend': 'Versioned assessment recommendations',
            'GET /api/info': 'API information'
        },
        'recommendation_details': {
            'min_recommendations': 5,
            'max_recommendations': 10,
            'supported_input': ['natural language query', 'job description text', 'URL to JD']
        }
    }
    
    return jsonify(info), 200


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))
    
    # Run the app
    logger.info(f"Starting Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
