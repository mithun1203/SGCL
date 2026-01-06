"""
Test ConceptNet API Connection
==============================

Quick script to verify ConceptNet API is working.
"""

import sys
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sid.conceptnet_client import ConceptNetClient, ConceptNetConfig

def test_api_connection():
    """Test basic ConceptNet API connection."""
    print("="*70)
    print("ConceptNet API Connection Test")
    print("="*70)
    
    # Test 1: Direct API request
    print("\n[Test 1] Direct API Request")
    api_url = "https://api.conceptnet.io"
    test_endpoint = f"{api_url}/c/en/bird"
    
    try:
        response = requests.get(test_endpoint, timeout=10)
        print(f"✓ URL: {test_endpoint}")
        print(f"✓ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ API is WORKING")
            data = response.json()
            print(f"✓ Received data with {len(data.get('edges', []))} edges")
        elif response.status_code == 502:
            print("✗ API returned 502 Bad Gateway (server error)")
            print("  This is a temporary ConceptNet server issue")
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection failed: {e}")
        print("  Check your internet connection")
    
    # Test 2: Using ConceptNetClient
    print("\n[Test 2] ConceptNetClient Query")
    try:
        config = ConceptNetConfig(offline_only=False)
        client = ConceptNetClient(config=config)
        
        print("✓ Client initialized")
        
        # Query with API
        edges = client.get_edges_for_concept('bird')
        print(f"✓ Found {len(edges)} edges for 'bird'")
        
        if edges:
            print("\nSample edges:")
            for edge in edges[:5]:
                print(f"  - {edge.relation}: {edge.start} -> {edge.end} (weight: {edge.weight:.2f})")
        
    except Exception as e:
        print(f"✗ Client query failed: {e}")
    
    # Test 3: Offline mode (local KB)
    print("\n[Test 3] Offline Mode (Local KB)")
    try:
        config_offline = ConceptNetConfig(offline_only=True)
        client_offline = ConceptNetClient(config=config_offline)
        
        print("✓ Offline client initialized")
        
        # Query with local KB only
        edges_offline = client_offline.get_edges_for_concept('bird')
        print(f"✓ Found {len(edges_offline)} edges in local KB")
        
        if edges_offline:
            print("\nSample local KB edges:")
            for edge in edges_offline[:5]:
                print(f"  - {edge.relation}: {edge.start} -> {edge.end}")
        
    except Exception as e:
        print(f"✗ Offline query failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("ConceptNet API: Check output above")
    print("Local KB: Available as fallback")
    print("Recommendation: Use offline_only=True for reliability")
    print("="*70)

if __name__ == "__main__":
    test_api_connection()
