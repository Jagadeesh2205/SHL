"""
Generate predictions for test dataset
Outputs predictions in the required CSV format
"""

import sys
import os
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import RAGRecommender
from src.evaluator import RecommendationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_predictions(test_queries, output_file='predictions.csv', k=10):
    """
    Generate predictions for test queries
    
    Args:
        test_queries: List of test queries
        output_file: Output CSV file path
        k: Number of recommendations per query
    """
    logger.info(f"Generating predictions for {len(test_queries)} test queries...")
    
    # Initialize recommender
    recommender = RAGRecommender()
    
    # Generate predictions
    all_predictions = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"Processing query {i}/{len(test_queries)}: {query[:60]}...")
        
        try:
            recommendations = recommender.recommend(query, k=k)
            
            for rec in recommendations:
                all_predictions.append({
                    'query': query,
                    'assessment_url': rec['url']
                })
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_predictions)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(all_predictions)} predictions to {output_file}")
    
    return df


def main():
    """Main function"""
    # Load test queries
    evaluator = RecommendationEvaluator()
    test_queries = evaluator.load_test_data('Gen_AI Dataset.xlsx')
    
    if not test_queries:
        logger.error("No test queries found. Make sure Gen_AI Dataset.xlsx is in the root directory.")
        return
    
    logger.info(f"Loaded {len(test_queries)} test queries")
    
    # Generate predictions
    df = generate_predictions(test_queries, output_file='predictions.csv', k=10)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Prediction Generation Complete!")
    print(f"{'='*80}")
    print(f"Total queries: {len(test_queries)}")
    print(f"Total predictions: {len(df)}")
    print(f"\nOutput file: predictions.csv")
    print(f"\nSample predictions:")
    print(df.head(10))
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
