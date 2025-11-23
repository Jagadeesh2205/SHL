// API endpoint - change this if deploying to a different URL
const API_BASE_URL = window.location.origin;

// DOM elements
const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const resultsCount = document.getElementById('resultsCount');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');
const exampleChips = document.querySelectorAll('.example-chip');

// Event listeners
searchBtn.addEventListener('click', handleSearch);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        handleSearch();
    }
});

exampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
        queryInput.value = chip.textContent;
        handleSearch();
    });
});

// Main search handler
async function handleSearch() {
    const query = queryInput.value.trim();

    // Validate input
    if (!query) {
        showError('Please enter a query or job description');
        return;
    }

    if (query.length < 3) {
        showError('Query must be at least 3 characters long');
        return;
    }

    // Hide previous results/errors
    hideError();
    hideResults();

    // Show loading state
    setLoading(true);

    try {
        // Make API request
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get recommendations');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred. Please try again.');
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    // Support both old and new API response formats
    const recommendations = data.recommended_assessments || data.recommendations;

    if (!recommendations || recommendations.length === 0) {
        showError('No recommendations found for your query');
        return;
    }

    // Update results count
    resultsCount.textContent = `Showing ${recommendations.length} recommendations`;

    // Clear previous results
    resultsContainer.innerHTML = '';

    // Create result cards
    recommendations.forEach((rec, index) => {
        const card = createResultCard(rec, index + 1);
        resultsContainer.appendChild(card);
    });

    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create a result card
function createResultCard(rec, rank) {
    const card = document.createElement('div');
    card.className = 'result-card';

    // Support both old and new formats
    const name = rec.name || rec.assessment_name;
    const testType = Array.isArray(rec.test_type) ? rec.test_type[0] : rec.test_type;
    const testTypeLabel = getTestTypeLabel(testType);
    const duration = rec.duration ? `${rec.duration} min` : '';
    const adaptive = rec.adaptive_support || 'No';
    const remote = rec.remote_support || 'Yes';

    card.innerHTML = `
        <div class="result-header">
            <div class="result-rank">${rank}</div>
            <div class="result-info">
                <h3 class="result-title">${escapeHtml(name)}</h3>
                <div class="result-meta">
                    <span class="badge badge-type">${testTypeLabel}</span>
                    ${duration ? `<span class="badge badge-duration">Duration: ${duration}</span>` : ''}
                    <span class="badge badge-remote">Remote: ${remote}</span>
                    <span class="badge badge-adaptive">Adaptive: ${adaptive}</span>
                </div>
                ${rec.description ? `<p class="result-description">${escapeHtml(rec.description)}</p>` : ''}
                <a href="${escapeHtml(rec.url)}" target="_blank" class="result-link">
                    View Assessment →
                </a>
            </div>
        </div>
    `;

    return card;
}

// Get test type label
function getTestTypeLabel(testType) {
    // If it's already a full name, return it
    if (testType && testType.length > 2) {
        return testType;
    }
    
    // Otherwise map the code
    const labels = {
        'C': 'Ability & Aptitude',
        'P': 'Personality & Behavior',
        'K': 'Knowledge & Skills',
        'S': 'Simulations',
        'O': 'Other'
    };
    return labels[testType] || testType || 'Assessment';
}

// Show error message
function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

// Hide error message
function hideError() {
    errorSection.style.display = 'none';
}

// Hide results
function hideResults() {
    resultsSection.style.display = 'none';
}

// Set loading state
function setLoading(isLoading) {
    const btnText = searchBtn.querySelector('.btn-text');
    const spinner = searchBtn.querySelector('.loading-spinner');

    if (isLoading) {
        btnText.style.display = 'none';
        spinner.style.display = 'inline';
        searchBtn.disabled = true;
    } else {
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
        searchBtn.disabled = false;
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Check API health on page load
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.recommender_ready) {
            showError('System is initializing. Please run the setup scripts first.');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showError('Unable to connect to the API. Please ensure the server is running.');
    }
}

// Run health check on page load
window.addEventListener('DOMContentLoaded', checkApiHealth);
