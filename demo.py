#!/usr/bin/env python3
"""
Interactive Demo for Semantic Search Engine
============================================
Run this script to see the semantic search in action!
"""

from semantic_search import SemanticSearchEngine


def main():
    print("=" * 60)
    print("🔍 Semantic Search Engine - Interactive Demo")
    print("=" * 60)
    print()
    
    # Initialize engine
    engine = SemanticSearchEngine()
    
    # Sample knowledge base
    knowledge_base = [
        # Technology
        "Python is a versatile programming language used for web development, data science, and automation",
        "JavaScript is the language of the web, running in browsers and on servers with Node.js",
        "Machine learning algorithms learn patterns from data to make predictions",
        "Deep learning uses neural networks with multiple layers to process complex data",
        "Natural language processing enables computers to understand human language",
        
        # Science
        "Photosynthesis is the process by which plants convert sunlight into energy",
        "DNA contains the genetic instructions for all living organisms",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape",
        "Climate change is causing global temperatures to rise and weather patterns to shift",
        "Quantum mechanics describes the behavior of matter at the atomic scale",
        
        # Business
        "Startups often seek venture capital funding to scale their operations",
        "Marketing strategies help companies reach their target audience effectively",
        "Supply chain management optimizes the flow of goods from production to consumers",
        "Customer relationship management (CRM) systems track interactions with clients",
        "Agile methodology promotes iterative development and team collaboration",
        
        # Health
        "Regular exercise improves cardiovascular health and mental well-being",
        "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals",
        "Sleep is essential for memory consolidation and physical recovery",
        "Vaccines train the immune system to recognize and fight pathogens",
        "Meditation and mindfulness can reduce stress and improve focus",
    ]
    
    print(f"📚 Loading {len(knowledge_base)} documents into the search index...")
    print()
    engine.add_documents(knowledge_base)
    print()
    
    # Demo queries
    demo_queries = [
        "How do computers understand text?",
        "What makes startups successful?",
        "How can I improve my health?",
        "Tell me about space and the universe",
        "Programming languages for beginners"
    ]
    
    print("=" * 60)
    print("📊 Demo Searches")
    print("=" * 60)
    
    for query in demo_queries:
        print(f"\n🔎 Query: \"{query}\"")
        print("-" * 50)
        
        results = engine.search(query, top_k=3)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {doc[:70]}...")
    
    print("\n" + "=" * 60)
    print("🎮 Interactive Mode")
    print("=" * 60)
    print("Type your search queries below. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("🔍 Search: ").strip()
            
            if not query:
                continue
            if query.lower() in ('quit', 'exit', 'q'):
                print("\n👋 Thanks for trying Semantic Search Engine!")
                break
            
            results = engine.search(query, top_k=5)
            
            if not results:
                print("  No results found.\n")
                continue
            
            print()
            for i, (doc, score) in enumerate(results, 1):
                # Color-code by relevance
                if score >= 0.7:
                    marker = "🟢"
                elif score >= 0.5:
                    marker = "🟡"
                else:
                    marker = "🔴"
                print(f"  {marker} [{score:.3f}] {doc}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()
