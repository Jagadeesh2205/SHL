"""
Evaluation metrics for the recommendation system
Implements Mean Recall@K and other metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluator for assessment recommendation system"""
    
    def __init__(self):
        self.results = []
    
    def recall_at_k(self, predicted: List[str], relevant: List[str], k: int = 10) -> float:
        """
        Calculate Recall@K for a single query
        
        Args:
            predicted: List of predicted assessment URLs
            relevant: List of relevant (ground truth) assessment URLs
            k: Number of top predictions to consider
            
        Returns:
            Recall@K score (0 to 1)
        """
        if not relevant:
            return 0.0
        
        # Take top k predictions
        predicted_k = predicted[:k]
        
        # Count how many relevant items are in top k
        hits = len(set(predicted_k) & set(relevant))
        
        # Recall = hits / total relevant
        recall = hits / len(relevant)
        
        return recall
    
    def mean_recall_at_k(self, predictions: Dict[str, List[str]], 
                        ground_truth: Dict[str, List[str]], k: int = 10) -> float:
        """
        Calculate Mean Recall@K across all queries
        
        Args:
            predictions: Dict mapping query to list of predicted URLs
            ground_truth: Dict mapping query to list of relevant URLs
            k: Number of top predictions to consider
            
        Returns:
            Mean Recall@K score (0 to 1)
        """
        recalls = []
        
        for query in ground_truth:
            if query not in predictions:
                logger.warning(f"Query not in predictions: {query}")
                recalls.append(0.0)
                continue
            
            predicted = predictions[query]
            relevant = ground_truth[query]
            
            recall = self.recall_at_k(predicted, relevant, k)
            recalls.append(recall)
            
            logger.debug(f"Query: {query[:50]}... | Recall@{k}: {recall:.3f}")
        
        mean_recall = np.mean(recalls) if recalls else 0.0
        return mean_recall
    
    def evaluate_diversity(self, predictions: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        Evaluate diversity of recommendations (test type distribution)
        
        Args:
            predictions: Dict mapping query to list of assessment dicts
            
        Returns:
            Dict with diversity metrics
        """
        diversity_scores = []
        
        for query, assessments in predictions.items():
            test_types = [a.get('test_type', 'O') for a in assessments]
            unique_types = len(set(test_types))
            total_types = len(test_types)
            
            # Diversity = unique types / total types
            diversity = unique_types / total_types if total_types > 0 else 0
            diversity_scores.append(diversity)
        
        return {
            'mean_diversity': np.mean(diversity_scores),
            'std_diversity': np.std(diversity_scores),
            'min_diversity': np.min(diversity_scores),
            'max_diversity': np.max(diversity_scores)
        }
    
    def evaluate_from_csv(self, predictions_csv: str, ground_truth_csv: str, k: int = 10) -> Dict:
        """
        Evaluate predictions from CSV files
        
        Args:
            predictions_csv: Path to predictions CSV (query, assessment_url)
            ground_truth_csv: Path to ground truth CSV (query, assessment_url)
            k: Number of top predictions to consider
            
        Returns:
            Dict with evaluation metrics
        """
        # Load predictions
        pred_df = pd.read_csv(predictions_csv)
        
        # Load ground truth
        gt_df = pd.read_csv(ground_truth_csv)
        
        # Group by query
        predictions = {}
        for query, group in pred_df.groupby('query'):
            predictions[query] = group['assessment_url'].tolist()
        
        ground_truth = {}
        for query, group in gt_df.groupby('query'):
            ground_truth[query] = group['assessment_url'].tolist()
        
        # Calculate metrics
        mean_recall = self.mean_recall_at_k(predictions, ground_truth, k)
        
        # Per-query recall
        per_query_recalls = {}
        for query in ground_truth:
            if query in predictions:
                recall = self.recall_at_k(predictions[query], ground_truth[query], k)
                per_query_recalls[query] = recall
        
        results = {
            'mean_recall_at_k': mean_recall,
            'k': k,
            'num_queries': len(ground_truth),
            'per_query_recalls': per_query_recalls
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """Print a formatted evaluation report"""
        print(f"\n{'='*80}")
        print("EVALUATION REPORT")
        print(f"{'='*80}\n")
        
        print(f"Mean Recall@{results['k']}: {results['mean_recall_at_k']:.4f}")
        print(f"Number of queries: {results['num_queries']}")
        print()
        
        if 'per_query_recalls' in results:
            print("Per-Query Results:")
            print("-" * 80)
            for query, recall in results['per_query_recalls'].items():
                print(f"Query: {query[:60]}...")
                print(f"  Recall@{results['k']}: {recall:.4f}")
                print()
        
        print(f"{'='*80}\n")
    
    def load_train_data(self, filepath: str = 'Gen_AI Dataset.xlsx') -> Tuple[Dict, Dict]:
        """
        Load training data from Excel file
        
        Args:
            filepath: Path to Excel file
            
        Returns:
            Tuple of (queries dict, ground_truth dict)
        """
        try:
            # Try to read different sheet names
            try:
                df = pd.read_excel(filepath, sheet_name='Train')
            except:
                try:
                    df = pd.read_excel(filepath, sheet_name='train')
                except:
                    df = pd.read_excel(filepath, sheet_name=0)
            
            logger.info(f"Loaded {len(df)} rows from training data")
            
            # Group by query
            queries = {}
            ground_truth = {}
            
            for query, group in df.groupby(df.columns[0]):  # First column is query
                queries[query] = query
                ground_truth[query] = group[df.columns[1]].tolist()  # Second column is URLs
            
            logger.info(f"Loaded {len(queries)} unique queries")
            return queries, ground_truth
            
        except Exception as e:
            logger.error(f"Error loading train data: {e}")
            return {}, {}
    
    def load_test_data(self, filepath: str = 'Gen_AI Dataset.xlsx') -> List[str]:
        """
        Load test queries from Excel file
        
        Args:
            filepath: Path to Excel file
            
        Returns:
            List of test queries
        """
        try:
            # Try to read different sheet names
            try:
                df = pd.read_excel(filepath, sheet_name='Test')
            except:
                try:
                    df = pd.read_excel(filepath, sheet_name='test')
                except:
                    df = pd.read_excel(filepath, sheet_name=1)
            
            queries = df[df.columns[0]].tolist()  # First column is query
            logger.info(f"Loaded {len(queries)} test queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return []


def main():
    """Test evaluator with sample data"""
    evaluator = RecommendationEvaluator()
    
    # Example evaluation
    predictions = {
        "Query 1": ["url1", "url2", "url3", "url4", "url5"],
        "Query 2": ["url6", "url7", "url8", "url9", "url10"]
    }
    
    ground_truth = {
        "Query 1": ["url1", "url2", "url11", "url12"],
        "Query 2": ["url7", "url9", "url13", "url14", "url15"]
    }
    
    mean_recall = evaluator.mean_recall_at_k(predictions, ground_truth, k=10)
    
    print(f"Mean Recall@10: {mean_recall:.4f}")
    
    # Test loading from Excel
    print("\nTrying to load dataset...")
    queries, gt = evaluator.load_train_data('Gen_AI Dataset.xlsx')
    
    if queries:
        print(f"Loaded {len(queries)} training queries")
        print("\nSample queries:")
        for i, query in enumerate(list(queries.keys())[:3], 1):
            print(f"{i}. {query}")
    
    test_queries = evaluator.load_test_data('Gen_AI Dataset.xlsx')
    if test_queries:
        print(f"\nLoaded {len(test_queries)} test queries")


if __name__ == "__main__":
    main()
