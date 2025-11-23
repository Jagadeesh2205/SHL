"""
Complete end-to-end runner for the SHL Assessment Recommendation System
This script runs all necessary steps in sequence
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(text)
    print(f"{'='*80}\n")


def check_environment():
    """Check if environment is set up correctly"""
    print_header("Step 0: Environment Check")
    
    # Check for .env file
    if not os.path.exists('.env'):
        logger.warning(".env file not found. Creating from template...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✓ Created .env file")
            print("⚠ Please edit .env and add your GOOGLE_API_KEY")
            print("  You can get a free API key from: https://ai.google.dev/")
        else:
            logger.error("No .env.example found")
    else:
        print("✓ .env file exists")
    
    # Check for dataset
    if not os.path.exists('Gen_AI Dataset.xlsx'):
        logger.warning("Gen_AI Dataset.xlsx not found in root directory")
        print("⚠ Please ensure Gen_AI Dataset.xlsx is in the root directory")
    else:
        print("✓ Dataset file exists")
    
    print("\n✓ Environment check complete")


def run_scraper():
    """Run the web scraper"""
    print_header("Step 1: Web Scraping")
    
    from src.scraper import SHLScraper
    
    scraper = SHLScraper()
    
    # Check if data already exists
    if os.path.exists('data/scraped_data.json'):
        response = input("Scraped data already exists. Re-scrape? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipping scraping, using existing data")
            scraper.load_from_json()
            print(f"✓ Loaded {len(scraper.assessments)} existing assessments")
            return
    
    # Scrape
    assessments = scraper.scrape_catalog()
    scraper.save_to_json()
    
    print(f"✓ Scraped and saved {len(assessments)} assessments")


def generate_embeddings():
    """Generate embeddings and build FAISS index"""
    print_header("Step 2: Generate Embeddings")
    
    from src.embeddings import EmbeddingManager
    
    manager = EmbeddingManager()
    
    # Check if embeddings already exist
    if os.path.exists('data/embeddings/faiss.index'):
        response = input("Embeddings already exist. Regenerate? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipping embedding generation, using existing embeddings")
            manager.load_embeddings()
            print(f"✓ Loaded existing embeddings for {len(manager.assessments)} assessments")
            return
    
    # Generate
    manager.load_assessments()
    manager.generate_embeddings()
    manager.build_faiss_index()
    manager.save_embeddings()
    
    print(f"✓ Generated and saved embeddings for {len(manager.assessments)} assessments")


def test_recommender():
    """Test the recommendation system"""
    print_header("Step 3: Test Recommendation System")
    
    from src.recommender import RAGRecommender
    
    print("Initializing recommender...")
    recommender = RAGRecommender()
    
    test_queries = [
        "Java developer with good communication skills",
        "Python programming assessment",
        "Leadership skills"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        recommendations = recommender.recommend(query, k=5)
        
        print(f"Top 5 recommendations:")
        for rec in recommendations:
            print(f"  {rec['rank']}. {rec['assessment_name']} (Type: {rec['test_type']}, Score: {rec.get('relevance_score', 0):.3f})")
    
    print("\n✓ Recommendation system test complete")


def generate_test_predictions():
    """Generate predictions for test dataset"""
    print_header("Step 4: Generate Test Predictions")
    
    response = input("Generate predictions for test dataset? (Y/n): ").strip().lower()
    if response == 'n':
        print("Skipping prediction generation")
        return
    
    from src.generate_predictions import generate_predictions
    from src.evaluator import RecommendationEvaluator
    
    evaluator = RecommendationEvaluator()
    test_queries = evaluator.load_test_data('Gen_AI Dataset.xlsx')
    
    if not test_queries:
        logger.warning("Could not load test queries. Skipping prediction generation.")
        return
    
    df = generate_predictions(test_queries, output_file='predictions.csv', k=10)
    print(f"✓ Generated {len(df)} predictions and saved to predictions.csv")


def start_api():
    """Start the Flask API server"""
    print_header("Step 5: Start API Server")
    
    response = input("Start the API server? (Y/n): ").strip().lower()
    if response == 'n':
        print("Skipping API server start")
        return
    
    print("\nStarting Flask API server...")
    print("The API will be available at: http://localhost:5000")
    print("The web interface will be at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    import subprocess
    subprocess.run([sys.executable, "api/app.py"])


def main():
    """Main execution function"""
    print_header("SHL Assessment Recommendation System - Complete Setup")
    
    print("This script will:")
    print("  1. Check environment")
    print("  2. Scrape SHL catalog (or use existing data)")
    print("  3. Generate embeddings (or use existing)")
    print("  4. Test the recommendation system")
    print("  5. Generate predictions for test dataset")
    print("  6. Start the API server")
    
    response = input("\nContinue? (Y/n): ").strip().lower()
    if response == 'n':
        print("Aborted")
        return
    
    try:
        # Run all steps
        check_environment()
        run_scraper()
        generate_embeddings()
        test_recommender()
        generate_test_predictions()
        start_api()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1
    
    print_header("Setup Complete!")
    print("Your SHL Assessment Recommendation System is ready to use!")
    print("\nTo restart the API later, run: python api/app.py")
    print("To generate new predictions, run: python src/generate_predictions.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
