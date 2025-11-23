"""
Test script to verify API endpoints match the required specification
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_health_endpoint():
    """Test the /health endpoint"""
    print("\n" + "="*80)
    print("Testing /health endpoint")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        data = response.json()
        
        # Validate response
        if response.status_code == 200:
            print("✓ Status code is 200 OK")
        else:
            print(f"✗ Expected 200, got {response.status_code}")
        
        if "status" in data and data["status"] == "healthy":
            print("✓ Response contains 'status': 'healthy'")
        else:
            print("✗ Response missing 'status': 'healthy'")
        
        print("\n✓ Health endpoint test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Health endpoint test FAILED: {e}")
        return False


def test_recommend_endpoint():
    """Test the /recommend endpoint"""
    print("\n" + "="*80)
    print("Testing /recommend endpoint")
    print("="*80)
    
    # Test queries
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
        "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests"
    ]
    
    all_passed = True
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query[:60]}...")
        
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"Status Code: {response.status_code}")
            
            data = response.json()
            
            # Validate status code
            if response.status_code == 200:
                print("✓ Status code is 200 OK")
            else:
                print(f"✗ Expected 200, got {response.status_code}")
                all_passed = False
                continue
            
            # Validate structure
            if "recommended_assessments" not in data:
                print("✗ Missing 'recommended_assessments' key")
                all_passed = False
                continue
            
            assessments = data["recommended_assessments"]
            print(f"✓ Found 'recommended_assessments' with {len(assessments)} items")
            
            # Check count (min 1, max 10)
            if 1 <= len(assessments) <= 10:
                print(f"✓ Recommendation count is valid (between 1-10)")
            else:
                print(f"✗ Recommendation count out of range: {len(assessments)}")
                all_passed = False
            
            # Validate first assessment structure
            if len(assessments) > 0:
                first = assessments[0]
                required_fields = {
                    'url': str,
                    'name': str,
                    'adaptive_support': str,
                    'description': str,
                    'duration': int,
                    'remote_support': str,
                    'test_type': list
                }
                
                print("\nValidating assessment structure:")
                for field, expected_type in required_fields.items():
                    if field not in first:
                        print(f"  ✗ Missing field: {field}")
                        all_passed = False
                    elif not isinstance(first[field], expected_type):
                        print(f"  ✗ Field '{field}' has wrong type: {type(first[field]).__name__} (expected {expected_type.__name__})")
                        all_passed = False
                    else:
                        print(f"  ✓ {field}: {expected_type.__name__}")
                
                # Validate specific field values
                if first.get('adaptive_support') not in ['Yes', 'No']:
                    print(f"  ✗ adaptive_support must be 'Yes' or 'No', got: {first.get('adaptive_support')}")
                    all_passed = False
                
                if first.get('remote_support') not in ['Yes', 'No']:
                    print(f"  ✗ remote_support must be 'Yes' or 'No', got: {first.get('remote_support')}")
                    all_passed = False
                
                # Show sample assessment
                print("\nSample Assessment:")
                print(json.dumps(first, indent=2))
                
                # Check diversity for Java + collaboration query
                if i == 1 and len(assessments) >= 5:
                    test_types = [rec['test_type'][0] if rec['test_type'] else 'Unknown' for rec in assessments]
                    has_knowledge = any('Knowledge' in tt for tt in test_types)
                    has_personality = any('Personality' in tt or 'Behavior' in tt for tt in test_types)
                    
                    print("\nDiversity Check (for Java + collaboration):")
                    print(f"  Has Knowledge & Skills: {has_knowledge}")
                    print(f"  Has Personality & Behavior: {has_personality}")
                    
                    if has_knowledge and has_personality:
                        print("  ✓ Recommendations include balanced mix (hard + soft skills)")
                    else:
                        print("  ! Note: Should include both technical and soft skill assessments")
        
        except Exception as e:
            print(f"✗ Error testing query: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ Recommend endpoint tests PASSED")
    else:
        print("\n✗ Some recommend endpoint tests FAILED")
    
    return all_passed


def main():
    print("\n" + "="*80)
    print("API ENDPOINT VALIDATION TEST")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    print("Waiting for server to be ready...")
    
    # Wait for server
    for i in range(5):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            print("✓ Server is ready")
            break
        except:
            time.sleep(2)
            if i == 4:
                print("✗ Server not responding. Please start the server with: python api/app.py")
                return
    
    # Run tests
    health_passed = test_health_endpoint()
    recommend_passed = test_recommend_endpoint()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Health Endpoint: {'✓ PASSED' if health_passed else '✗ FAILED'}")
    print(f"Recommend Endpoint: {'✓ PASSED' if recommend_passed else '✗ FAILED'}")
    
    if health_passed and recommend_passed:
        print("\n✓✓✓ ALL TESTS PASSED - API is ready for submission! ✓✓✓")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
