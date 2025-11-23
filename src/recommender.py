"""
RAG-based recommendation engine using LangChain and Google Gemini
"""

import os
from typing import List, Dict
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini not available. Install with: pip install google-generativeai langchain-google-genai")

from src.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRecommender:
    """RAG-based Assessment Recommender using Gemini LLM"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize RAG recommender
        
        Args:
            api_key: Google Gemini API key (or uses GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
        
        # Load embeddings and index
        if not self.embedding_manager.load_embeddings():
            logger.warning("Failed to load embeddings. Generating new ones...")
            self.embedding_manager.load_assessments()
            self.embedding_manager.generate_embeddings()
            self.embedding_manager.build_faiss_index()
            self.embedding_manager.save_embeddings()
        
        # Initialize LLM if API key is available
        self.llm = None
        self.chain = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=self.api_key,
                    temperature=0.2,
                    convert_system_message_to_human=True
                )
                self._setup_chain()
                logger.info("Initialized Gemini LLM for RAG")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        else:
            logger.warning("Gemini not available. Using embeddings-only mode.")
    
    def _setup_chain(self):
        """Setup LangChain prompt and chain"""
        template = """You are an expert HR assessment consultant specializing in SHL assessments.

Given a job requirement or query, analyze what skills and competencies are needed and recommend the most relevant assessments.

Query: {query}

Retrieved Assessments (from vector search):
{retrieved_assessments}

Your task:
1. Analyze the query to identify required skills (technical, cognitive, personality, behavioral)
2. Review the retrieved assessments and their relevance
3. Select the top 10 most relevant assessments
4. Ensure a balanced mix if the query requires multiple skill types (e.g., technical + communication)
5. Rank them by relevance

Provide your recommendations as a numbered list with brief justification for each.

Recommendations:"""

        prompt = PromptTemplate(
            input_variables=["query", "retrieved_assessments"],
            template=template
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
    
    def recommend(self, query: str, k: int = 10, use_llm: bool = True) -> List[Dict]:
        """
        Generate assessment recommendations
        
        Args:
            query: User query or job description
            k: Number of recommendations (max 10)
            use_llm: Whether to use LLM for reranking (if available)
            
        Returns:
            List of recommended assessments
        """
        k = min(k, 10)  # Cap at 10 as per requirements
        k = max(k, 5)   # Minimum 5 as per requirements
        
        logger.info(f"Generating recommendations for query: {query[:50]}...")
        
        # Step 1: Retrieve candidates using vector search
        candidates = self.embedding_manager.search_with_diversity(query, k=k)
        
        if not candidates:
            logger.warning("No candidates found")
            return []
        
        # Step 2: Use LLM for reranking and refinement (if available)
        if use_llm and self.llm and self.chain:
            try:
                recommendations = self._llm_rerank(query, candidates, k)
            except Exception as e:
                logger.error(f"LLM reranking failed: {e}")
                recommendations = [assessment for assessment, score in candidates[:k]]
        else:
            # Fallback: use vector search results directly
            recommendations = [assessment for assessment, score in candidates[:k]]
        
        # Ensure we have between 5-10 results
        recommendations = recommendations[:k]
        
        # Add ranking
        for i, rec in enumerate(recommendations, 1):
            rec['rank'] = i
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _llm_rerank(self, query: str, candidates: List[tuple], k: int) -> List[Dict]:
        """
        Use LLM to rerank and refine recommendations
        
        Args:
            query: User query
            candidates: List of (assessment, score) tuples
            k: Number of results to return
            
        Returns:
            Reranked list of assessments
        """
        # Format candidates for LLM
        retrieved_text = ""
        for i, (assessment, score) in enumerate(candidates, 1):
            retrieved_text += f"{i}. {assessment['assessment_name']}\n"
            retrieved_text += f"   Description: {assessment.get('description', 'N/A')}\n"
            retrieved_text += f"   Type: {assessment.get('test_type', 'N/A')}\n"
            retrieved_text += f"   Relevance Score: {score:.3f}\n\n"
        
        # Run LLM chain
        try:
            response = self.chain.run(
                query=query,
                retrieved_assessments=retrieved_text
            )
            
            # Parse LLM response to extract recommended assessments
            # The LLM should maintain the order of relevance
            recommended = []
            
            # Try to match assessment names from response
            for assessment, score in candidates:
                name = assessment['assessment_name']
                if name in response:
                    recommended.append(assessment)
            
            # If parsing fails, fall back to original order
            if len(recommended) < k:
                logger.warning("LLM parsing incomplete, using original ranking")
                recommended = [assessment for assessment, score in candidates[:k]]
            
            return recommended[:k]
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {e}")
            return [assessment for assessment, score in candidates[:k]]
    
    def recommend_from_jd_text(self, jd_text: str, k: int = 10) -> List[Dict]:
        """
        Recommend assessments from job description text
        
        Args:
            jd_text: Job description text
            k: Number of recommendations
            
        Returns:
            List of recommended assessments
        """
        # Extract key requirements from JD
        if self.llm:
            try:
                extraction_prompt = f"""Extract the key skills, competencies, and requirements from this job description:

{jd_text}

List the main technical skills, soft skills, and any other relevant requirements."""
                
                # Simple extraction (you could use a more sophisticated approach)
                query = jd_text[:1000]  # Use first 1000 chars
            except Exception as e:
                logger.error(f"JD extraction failed: {e}")
                query = jd_text[:1000]
        else:
            query = jd_text[:1000]
        
        return self.recommend(query, k=k)
    
    def get_assessment_details(self, assessment_url: str) -> Dict:
        """
        Get details of a specific assessment
        
        Args:
            assessment_url: URL of the assessment
            
        Returns:
            Assessment details dict
        """
        for assessment in self.embedding_manager.assessments:
            if assessment['url'] == assessment_url:
                return assessment
        return None


def main():
    """Test the recommender"""
    print("Initializing RAG Recommender...")
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("\nWarning: GOOGLE_API_KEY not found in environment")
        print("Set it in .env file for LLM-enhanced recommendations")
        print("Continuing with embeddings-only mode...\n")
    
    # Initialize recommender
    recommender = RAGRecommender(api_key=api_key)
    
    # Test queries
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
        "Need assessment for analyst role with cognitive and personality tests",
        "Sales manager with leadership and communication skills",
        "Entry-level software engineer position"
    ]
    
    print(f"\n{'='*80}")
    print("Testing RAG Recommender")
    print(f"{'='*80}\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 80)
        
        recommendations = recommender.recommend(query, k=10)
        
        print(f"\nTop {len(recommendations)} Recommendations:\n")
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['assessment_name']}")
            print(f"   Type: {rec['test_type']} | Category: {rec.get('category', 'N/A')}")
            print(f"   Score: {rec.get('relevance_score', 0):.3f}")
            print(f"   URL: {rec['url']}")
            print()
        
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
