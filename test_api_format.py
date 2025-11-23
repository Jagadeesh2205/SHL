"""
Test script to verify the API response format matches requirements
"""

import requests
import json
import time

# Wait for server to start
print("Waiting for server to start...")
time.sleep(10)

# Test query
query = "Need a Java developer who is good in collaborating with external teams and stakeholders."

print(f"\n{'='*80}")
print(f"Testing API Response Format")
print(f"{'='*80}\n")
print(f"Query: {query}\n")

try:
    # Make API request
    response = requests.post(
        "http://localhost:5000/recommend",
        json={"query": query},
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"\nResponse Structure:")
        print(json.dumps(data, indent=2)[:500])
        
        # Verify structure
        if "recommended_assessments" in data:
            assessments = data["recommended_assessments"]
            print(f"\n✓ Found 'recommended_assessments' key")
            print(f"✓ Number of recommendations: {len(assessments)}")
            
            if len(assessments) > 0:
                first = assessments[0]
                print(f"\n✓ First recommendation structure:")
                print(json.dumps(first, indent=2))
                
                # Check required fields
                required_fields = ['url', 'name', 'adaptive_support', 'description', 'duration', 'remote_support', 'test_type']
                missing = [f for f in required_fields if f not in first]
                
                if missing:
                    print(f"\n✗ Missing fields: {missing}")
                else:
                    print(f"\n✓ All required fields present!")
                    
                    # Check test_type is a list
                    if isinstance(first['test_type'], list):
                        print(f"✓ test_type is a list: {first['test_type']}")
                    else:
                        print(f"✗ test_type should be a list, got: {type(first['test_type'])}")
                
                # Show diversity in recommendations
                print(f"\n{'='*80}")
                print(f"Recommendation Diversity:")
                print(f"{'='*80}\n")
                
                test_types = {}
                for rec in assessments[:10]:
                    test_type = rec['test_type'][0] if rec['test_type'] else 'Unknown'
                    test_types[test_type] = test_types.get(test_type, 0) + 1
                    print(f"{rec['name'][:50]:50} | {test_type}")
                
                print(f"\nTest Type Distribution:")
                for tt, count in test_types.items():
                    print(f"  {tt}: {count}")
        else:
            print(f"✗ Missing 'recommended_assessments' key")
            print(f"Available keys: {list(data.keys())}")
    else:
        print(f"Error: Status code {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print(f"Test Complete")
print(f"{'='*80}\n")
