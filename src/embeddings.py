"""
Embedding generation and vector database management
Uses sentence-transformers for embeddings and FAISS for vector search
"""

import json
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings and vector search for assessments"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.assessments = []
        self.embeddings = None
        self.index = None
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        logger.info(f"Initialized embedding manager with model: {model_name}")
    
    def load_assessments(self, filepath: str = 'data/scraped_data.json') -> List[Dict]:
        """Load assessments from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            logger.info(f"Loaded {len(self.assessments)} assessments")
            return self.assessments
        except Exception as e:
            logger.error(f"Error loading assessments: {e}")
            return []
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all assessments
        Combines assessment name, description, and category for rich representation
        """
        if not self.assessments:
            logger.error("No assessments loaded")
            return None
        
        logger.info("Generating embeddings...")
        
        # Create text representations
        texts = []
        for assessment in self.assessments:
            text = f"{assessment['assessment_name']} {assessment.get('description', '')} "
            text += f"{assessment.get('category', '')} {assessment.get('test_type', '')}"
            texts.append(text.strip())
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if self.embeddings is None:
            logger.error("No embeddings available. Generate embeddings first.")
            return
        
        logger.info("Building FAISS index...")
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Search for most relevant assessments
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (assessment, score) tuples
        """
        if self.index is None:
            logger.error("FAISS index not built")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.assessments):
                assessment = self.assessments[idx].copy()
                assessment['relevance_score'] = float(score)
                results.append((assessment, float(score)))
        
        return results
    
    def search_with_diversity(self, query: str, k: int = 10, 
                            diversity_weight: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Search with diversity to ensure balanced test types
        
        Args:
            query: Search query
            k: Number of results to return
            diversity_weight: Weight for diversity (0-1)
            
        Returns:
            List of (assessment, score) tuples with diverse test types
        """
        # Get more candidates than needed
        candidates = self.search(query, k * 3)
        
        if not candidates:
            return []
        
        # Group by test type
        by_type = {}
        for assessment, score in candidates:
            test_type = assessment.get('test_type', 'O')
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append((assessment, score))
        
        # Determine if query requires multiple types
        query_lower = query.lower()
        requires_multiple = any(word in query_lower for word in [
            'and', 'also', 'both', 'along with', 'as well as',
            'good communication', 'soft skill', 'collaborate', 'team'
        ])
        
        # Select results
        results = []
        
        if requires_multiple and len(by_type) > 1:
            # Balanced selection from different types
            types_to_include = []
            
            # Prioritize K and P types for technical + soft skill queries
            if 'K' in by_type:
                types_to_include.append('K')
            if 'P' in by_type:
                types_to_include.append('P')
            if 'C' in by_type and len(types_to_include) < 2:
                types_to_include.append('C')
            if 'S' in by_type and len(types_to_include) < 2:
                types_to_include.append('S')
            
            # Add remaining types if needed
            for test_type in by_type:
                if test_type not in types_to_include:
                    types_to_include.append(test_type)
            
            # Distribute k results across types
            per_type = k // max(len(types_to_include), 1)
            remainder = k % max(len(types_to_include), 1)
            
            for i, test_type in enumerate(types_to_include):
                count = per_type + (1 if i < remainder else 0)
                results.extend(by_type[test_type][:count])
            
            # Sort by score and take top k
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:k]
        else:
            # Simple top-k by score
            results = candidates[:k]
        
        return results
    
    def save_embeddings(self, embeddings_dir: str = 'data/embeddings'):
        """Save embeddings and index to disk"""
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
        np.save(embeddings_path, self.embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save FAISS index
        index_path = os.path.join(embeddings_dir, 'faiss.index')
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save assessments with metadata
        metadata_path = os.path.join(embeddings_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'assessments': self.assessments,
                'model_name': self.model_name,
                'dimension': self.dimension
            }, f)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_embeddings(self, embeddings_dir: str = 'data/embeddings'):
        """Load embeddings and index from disk"""
        try:
            # Load embeddings
            embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings from {embeddings_path}")
            
            # Load FAISS index
            index_path = os.path.join(embeddings_dir, 'faiss.index')
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
            
            # Load assessments
            metadata_path = os.path.join(embeddings_dir, 'metadata.pkl')
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.assessments = metadata['assessments']
                self.model_name = metadata['model_name']
                self.dimension = metadata['dimension']
            logger.info(f"Loaded {len(self.assessments)} assessments from metadata")
            
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False


def main():
    """Main function to generate and save embeddings"""
    # Initialize manager
    manager = EmbeddingManager()
    
    # Load assessments
    assessments = manager.load_assessments()
    
    if not assessments:
        print("Error: No assessments loaded. Run scraper.py first.")
        return
    
    # Generate embeddings
    embeddings = manager.generate_embeddings()
    
    if embeddings is None:
        print("Error: Failed to generate embeddings")
        return
    
    # Build FAISS index
    manager.build_faiss_index()
    
    # Save everything
    manager.save_embeddings()
    
    # Test search
    print(f"\n{'='*60}")
    print("Testing search functionality")
    print(f"{'='*60}")
    
    test_queries = [
        "Java developer with good communication skills",
        "Python programming assessment",
        "Leadership and management skills"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = manager.search_with_diversity(query, k=5)
        print(f"Top 5 results:")
        for i, (assessment, score) in enumerate(results, 1):
            print(f"  {i}. {assessment['assessment_name']} (Score: {score:.3f}, Type: {assessment['test_type']})")
    
    print(f"\n{'='*60}")
    print("Embeddings generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
