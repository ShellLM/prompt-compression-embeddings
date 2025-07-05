#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly.
"""

import llm
import sys
import os
import sqlite3

def test_llm_api():
    """Test if llm Python API is available."""
    try:
        # Test getting a model - using an available model
        model = llm.get_model("claude-3.7-sonnet")
        print(f"✓ LLM Python API available: {model.model_id}")
        return True
    except Exception as e:
        print(f"✗ LLM Python API not available: {e}")
        return False

def test_embeddings():
    """Test if embeddings are available."""
    try:
        # Use an available embedding model
        embedding_model = llm.get_embedding_model("text-embedding-3-small")
        result = embedding_model.embed("test")
        if result:
            print("✓ Embeddings working")
            return True
        else:
            print("✗ Embeddings returned empty result")
            return False
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False

def test_database_access():
    """Test if we can access the LLM logs database."""
    db_path = os.path.expanduser('~/.config/io.datasette.llm/logs.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM responses")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"✓ Database accessible with {count} responses")
        return True
    except Exception as e:
        print(f"✗ Database access failed: {e}")
        return False

def test_sample_data():
    """Test if our sample response IDs exist."""
    db_path = os.path.expanduser('~/.config/io.datasette.llm/logs.db')
    test_ids = ['01jz0cvyg7075fnqkkp2yb6jq7', '01jz09kjeqkg2ktdqedyych20e', 
                '01jz0jwxpe8tbytrmey4b15gxa', '01jz3a906ksjkh0fwnbx4c9wvn']
    
    try:
        conn = sqlite3.connect(db_path)
        placeholders = ','.join('?' for _ in test_ids)
        query = f"SELECT id FROM responses WHERE id IN ({placeholders})"
        cursor = conn.execute(query, test_ids)
        found_ids = [row[0] for row in cursor]
        conn.close()
        
        print(f"✓ Found {len(found_ids)}/{len(test_ids)} test response IDs")
        if len(found_ids) < len(test_ids):
            missing = set(test_ids) - set(found_ids)
            print(f"  Missing: {missing}")
        return len(found_ids) > 0
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def main():
    print("Testing prompt compression setup...\n")
    
    tests = [
        test_llm_api,
        test_database_access,
        test_sample_data,
        test_embeddings
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run compression tests.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the setup before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
