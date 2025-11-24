"""
Streamlit App for SHL Assessment Recommendation System
A simpler deployment-ready version
"""

import streamlit as st
import json
import numpy as np
import os

# Try importing faiss; provide graceful fallback if unavailable (e.g. Python 3.13 wheels missing)
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ModuleNotFoundError:
    FAISS_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 SHL Assessment Recommendation System")
st.markdown("**RAG-based Assessment Recommender** - Enter a job description or requirements to get assessment recommendations")

@st.cache_resource
def load_model():
    """Load sentence transformer model"""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        st.error(
            "Missing dependency: sentence-transformers. Add it to requirements.txt and redeploy. "
            "Recommended pins: sentence-transformers==2.2.2, torch==2.2.2 (Python 3.11)."
        )
        return None
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_assessments():
    """Load assessment data"""
    with open('data/scraped_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_embeddings():
    """Load embeddings and (optionally) FAISS index"""
    embeddings = np.load('data/embeddings/embeddings.npy')
    if FAISS_AVAILABLE:
        index = faiss.read_index('data/embeddings/faiss.index')
        return embeddings, index
    # Normalize embeddings for cosine similarity fallback
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-9, norms)
    embeddings_normed = embeddings / safe_norms
    return embeddings_normed, None

# Load resources
with st.spinner("Loading models and data..."):
    model = load_model()
    assessments = load_assessments()
    embeddings, index = load_embeddings()

if model is None:
    st.stop()

st.success(f"✅ Loaded {len(assessments)} assessments")

# Input
query = st.text_area(
    "Enter job description or requirements:",
    height=150,
    placeholder="Example: I need to hire a Java developer with strong communication skills and problem-solving abilities..."
)

# Number of recommendations
k = st.slider("Number of recommendations:", min_value=5, max_value=10, value=10)

# Recommend button
if st.button("Get Recommendations", type="primary"):
    if not query or len(query) < 3:
        st.error("Please enter a valid query (at least 3 characters)")
    else:
        with st.spinner("Generating recommendations..."):
            # Generate query embedding
            query_embedding = model.encode([query], normalize_embeddings=True).astype('float32')

            if FAISS_AVAILABLE and index is not None:
                # FAISS search
                scores, indices = index.search(query_embedding, k)
                # Collect recommendations
                recommendations = []
                for idx, score in zip(indices[0], scores[0]):
                    assessment = assessments[idx].copy()
                    assessment['relevance_score'] = float(score)
                    recommendations.append(assessment)
            else:
                # NumPy cosine similarity fallback
                # embeddings already normalized; query embedding normalized via model.encode(normalize_embeddings=True)
                sims = embeddings @ query_embedding[0]
                top_idx = np.argsort(-sims)[:k]
                recommendations = []
                for idx in top_idx:
                    assessment = assessments[idx].copy()
                    assessment['relevance_score'] = float(sims[idx])
                    recommendations.append(assessment)
            
            # Display results
            st.subheader(f"Top {k} Recommended Assessments")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"#{i} - {rec['assessment_name']} (Score: {rec['relevance_score']:.3f})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {rec.get('description', 'N/A')}")
                        st.markdown(f"**Category:** {rec.get('category', 'N/A')}")
                        st.markdown(f"**Test Type:** {rec.get('test_type', 'N/A')}")
                    
                    with col2:
                        st.metric("Relevance", f"{rec['relevance_score']:.1%}")
                        if rec.get('url'):
                            st.markdown(f"[View Assessment]({rec['url']})")
            
            # Download as JSON
            import json
            result_json = {
                'query': query,
                'recommended_assessments': recommendations
            }
            st.download_button(
                label="Download Results (JSON)",
                data=json.dumps(result_json, indent=2),
                file_name="recommendations.json",
                mime="application/json"
            )

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system uses:
    - **Sentence Transformers** for embeddings
    - **Vector Search** via FAISS (if available) or NumPy fallback
    - **SHL Product Catalog** (scraped data)
    
    ### How it works:
    1. Enter job requirements
    2. System converts text to embeddings
    3. Finds similar assessments using vector search
    4. Returns ranked recommendations
    """)
    
    st.header("Statistics")
    st.metric("Total Assessments", len(assessments))
    st.metric("Embedding Dimension", embeddings.shape[1])
    st.metric("Vector Backend", "FAISS" if FAISS_AVAILABLE else "NumPy Fallback")
    
    st.header("API Endpoint")
    st.code("POST /recommend", language="http")
    st.markdown("For API access, contact the administrator.")

# Footer
st.markdown("---")
st.markdown("SHL Assessment Recommendation System")
