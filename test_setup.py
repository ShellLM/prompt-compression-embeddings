#!/usr/bin/env python3
"""
Quick test to verify the setup is working correctly.
"""

import subprocess
import sys
import os
import sqlite3

def test_llm_cli():
    """Test if llm CLI is available."""
    try:
        result = subprocess.run(['llm', '--version'], capture_output=True, text=True)
        print(f"✓ LLM CLI available: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ LLM CLI not found")
        return False

def test_jina_embeddings():
    """Test if jina embeddings are available."""
    try:
        result = subprocess.run(['sh', '-c', 'echo "test" | llm embed -m jina-v4'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Jina embeddings working")
            return True
        else:
            print(f"✗ Jina embeddings error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Jina embeddings test failed: {e}")
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
        test_llm_cli,
        test_database_access,
        test_sample_data,
        test_jina_embeddings
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
