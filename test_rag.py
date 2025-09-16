#!/usr/bin/env python3
"""Test script for RAG system"""

from rag import initialize_rag_system, get_rag_response, is_explanation_question
import traceback

def test_rag_system():
    try:
        print('=== Testing RAG System ===')
        
        # Initialize RAG system
        print('1. Initializing RAG system...')
        rag = initialize_rag_system(force_reload=True)
        
        # Check collection stats
        stats = rag.get_collection_stats()
        print(f'2. Collection stats: {stats}')
        
        # Test explanation detection
        test_questions = [
            'what are the azure services',
            'explain azure services',
            'what is azure',
            'show me the cost of services',  # Should be SQL
            'list services by cost'  # Should be SQL
        ]
        
        print('3. Testing explanation detection:')
        for q in test_questions:
            is_explanation = is_explanation_question(q)
            print(f'   "{q}" -> RAG: {is_explanation}')
        
        # Test RAG response
        print('4. Testing RAG response for Azure services question...')
        response = get_rag_response('what are the azure services')
        print(f'   Response length: {len(response)} characters')
        print(f'   Response preview: {response[:200]}...')
        
        print('=== RAG Test Complete ===')
        
    except Exception as e:
        print(f'Error during RAG test: {e}')
        print(f'Traceback: {traceback.format_exc()}')

if __name__ == "__main__":
    test_rag_system()
