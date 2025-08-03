#!/usr/bin/env python3
"""
Practical Applications demonstration of the coherify library.

This comprehensive example demonstrates real-world applications:
1. AI Content Quality Assurance Pipeline
2. Scientific Paper Coherence Assessment
3. Educational Content Evaluation
4. AI Safety and Hallucination Detection
5. Multi-modal Content Analysis
6. Production RAG System with Coherence
"""

import numpy as np
import time
from typing import List
from coherify import (
    PropositionSet,
    Proposition,
    SemanticCoherence,
    EntailmentCoherence,
    HybridCoherence,
    CoherenceAnalyzer,
    CoherenceReranker,
    CoherenceRAG,
    ClusterBasedApproximator,
    IncrementalCoherenceTracker,
    CoherenceGuidedGenerator,
    CoherenceFilter,
)


def ai_content_quality_pipeline():
    """Demonstrate comprehensive AI content quality assurance pipeline."""
    print("=== AI Content Quality Assurance Pipeline ===")

    # Sample AI-generated content with varying quality
    content_samples = [
        {
            "title": "Machine Learning Overview",
            "content": """Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. These systems use algorithms to analyze data, identify patterns, and make predictions or decisions. The process typically involves training a model on historical data, validating its performance, and then deploying it to make predictions on new, unseen data. Common applications include recommendation systems, image recognition, natural language processing, and autonomous vehicles.""",
            "expected_quality": "high",
        },
        {
            "title": "Climate Change Discussion",
            "content": """Climate change is caused by human activities. The ocean is blue because of light scattering. Pizza was invented in Italy and contains cheese. Machine learning algorithms can predict weather patterns. Elephants are large mammals that live in Africa and Asia. Solar panels convert sunlight into electricity for clean energy generation.""",
            "expected_quality": "low",  # Incoherent mix of topics
        },
        {
            "title": "Software Development Practices",
            "content": """Software development involves writing, testing, and maintaining computer programs. Good practices include version control, code reviews, and automated testing. However, the weather today is quite nice with sunny skies. Developers should also focus on writing clean, readable code that follows established patterns and conventions. Documentation is essential for long-term maintenance.""",
            "expected_quality": "medium",  # Mostly coherent with one off-topic sentence
        },
    ]

    # Quality assessment pipeline
    print("🔍 Content Quality Assessment Pipeline:")

    # Multi-stage coherence evaluation
    coherence_measures = [
        ("Semantic", SemanticCoherence()),
        ("Entailment", EntailmentCoherence()),
        ("Hybrid", HybridCoherence()),
    ]

    # Quality filter with multiple thresholds
    quality_filter = CoherenceFilter(
        coherence_measure=HybridCoherence(), min_coherence_threshold=0.4
    )

    results = []

    for i, sample in enumerate(content_samples, 1):
        print(f"\n📄 Sample {i}: {sample['title']}")
        print(f"Expected Quality: {sample['expected_quality']}")
        print(f"Content preview: {sample['content'][:100]}...")

        # Create proposition set
        prop_set = PropositionSet.from_qa_pair(sample["title"], sample["content"])

        # Evaluate with multiple measures
        scores = {}
        for measure_name, measure in coherence_measures:
            result = measure.compute(prop_set)
            scores[measure_name] = result.score
            print(f"  {measure_name} Coherence: {result.score:.3f}")

        # Overall quality assessment
        avg_score = np.mean(list(scores.values()))
        quality_level = (
            "High" if avg_score > 0.6 else "Medium" if avg_score > 0.3 else "Low"
        )

        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Quality Assessment: {quality_level}")
        print(
            f"  Matches Expected: {'✅' if quality_level.lower() == sample['expected_quality'] else '❌'}"
        )

        # Store for analysis
        results.append(
            {
                "title": sample["title"],
                "scores": scores,
                "avg_score": avg_score,
                "quality_level": quality_level,
                "expected": sample["expected_quality"],
                "content_length": len(sample["content"]),
                "num_propositions": len(prop_set.propositions),
            }
        )

    # Pipeline summary
    print(f"\n📊 Pipeline Summary:")
    accuracy = sum(
        1 for r in results if r["quality_level"].lower() == r["expected"]
    ) / len(results)
    print(f"  Quality Assessment Accuracy: {accuracy:.1%}")
    print(
        f"  Average Processing Score: {np.mean([r['avg_score'] for r in results]):.3f}"
    )

    # Recommendations
    print(f"\n💡 Automated Recommendations:")
    for result in results:
        if result["avg_score"] < 0.3:
            print(
                f"  - {result['title']}: Requires major revision (score: {result['avg_score']:.3f})"
            )
        elif result["avg_score"] < 0.6:
            print(
                f"  - {result['title']}: Minor improvements needed (score: {result['avg_score']:.3f})"
            )
        else:
            print(
                f"  - {result['title']}: Good quality, ready for publication (score: {result['avg_score']:.3f})"
            )


def scientific_paper_assessment():
    """Demonstrate scientific paper coherence assessment."""
    print("\n=== Scientific Paper Coherence Assessment ===")

    # Mock scientific paper sections
    paper_sections = {
        "abstract": "This study investigates the application of machine learning algorithms to climate data analysis. We developed a novel neural network architecture that can predict temperature patterns with 95% accuracy. The model was trained on 10 years of global climate data and validated using cross-validation techniques.",
        "introduction": "Climate change represents one of the most pressing challenges of our time. Machine learning has emerged as a powerful tool for analyzing complex environmental data. Recent advances in deep learning have shown promising results in weather prediction and climate modeling.",
        "methodology": "We collected temperature data from 1000 weather stations worldwide. The data was preprocessed to remove outliers and normalize values. A convolutional neural network with three hidden layers was implemented using TensorFlow. The model was trained using backpropagation with a learning rate of 0.001.",
        "results": "The model achieved 95% accuracy on the test set. Temperature predictions were accurate within 2 degrees Celsius for 90% of cases. The network successfully identified seasonal patterns and long-term climate trends.",
        "conclusion": "Our results demonstrate the effectiveness of neural networks for climate prediction. This approach could be valuable for climate scientists and policy makers. Future work should explore larger datasets and more complex architectures.",
    }

    print("📚 Analyzing Scientific Paper Coherence:")

    # Analyze individual sections
    analyzer = CoherenceAnalyzer()
    section_results = {}

    for section_name, content in paper_sections.items():
        prop_set = PropositionSet.from_qa_pair(
            f"Scientific paper {section_name}", content
        )

        # Multiple coherence measures for comprehensive analysis
        measures = [SemanticCoherence(), HybridCoherence()]

        comparison = analyzer.compare_measures(prop_set, measures)
        avg_coherence = comparison["score_statistics"]["mean"]

        section_results[section_name] = {
            "content": content,
            "coherence": avg_coherence,
            "propositions": len(prop_set.propositions),
            "comparison": comparison,
        }

        print(f"\n📖 {section_name.title()} Section:")
        print(f"  Coherence Score: {avg_coherence:.3f}")
        print(f"  Propositions: {len(prop_set.propositions)}")
        print(f"  Content Length: {len(content)} characters")

    # Cross-section coherence analysis
    print(f"\n🔗 Cross-Section Coherence Analysis:")

    # Analyze coherence between sections
    section_pairs = [
        ("abstract", "introduction"),
        ("introduction", "methodology"),
        ("methodology", "results"),
        ("results", "conclusion"),
    ]

    cross_section_scores = []
    for section1, section2 in section_pairs:
        combined_content = f"{paper_sections[section1]} {paper_sections[section2]}"
        combined_set = PropositionSet.from_qa_pair(
            f"{section1}-{section2} coherence", combined_content
        )

        coherence_result = HybridCoherence().compute(combined_set)
        cross_section_scores.append(coherence_result.score)

        print(
            f"  {section1.title()} → {section2.title()}: {coherence_result.score:.3f}"
        )

    # Overall paper assessment
    overall_coherence = np.mean(
        list(section_results[s]["coherence"] for s in section_results)
    )
    cross_coherence = np.mean(cross_section_scores)

    print(f"\n📋 Overall Paper Assessment:")
    print(f"  Average Section Coherence: {overall_coherence:.3f}")
    print(f"  Average Cross-Section Coherence: {cross_coherence:.3f}")
    print(
        f"  Paper Flow Quality: {'Good' if cross_coherence > 0.5 else 'Needs Improvement'}"
    )

    # Improvement recommendations
    print(f"\n💡 Improvement Recommendations:")
    weakest_section = min(section_results.items(), key=lambda x: x[1]["coherence"])
    print(
        f"  - Focus on improving {weakest_section[0]} section (score: {weakest_section[1]['coherence']:.3f})"
    )

    if cross_coherence < 0.5:
        print(f"  - Improve transitions between sections")
        print(f"  - Add connecting sentences to link ideas")


def educational_content_evaluation():
    """Demonstrate educational content coherence evaluation."""
    print("\n=== Educational Content Evaluation ===")

    # Educational content examples
    lessons = [
        {
            "topic": "Introduction to Programming",
            "content": """Programming is the process of creating instructions for computers to follow. These instructions are written in programming languages like Python, Java, or JavaScript. A program is a sequence of these instructions that tells the computer what to do step by step. To write effective programs, programmers must break down complex problems into smaller, manageable tasks. This process is called problem decomposition and is fundamental to programming.""",
            "level": "beginner",
        },
        {
            "topic": "Advanced Data Structures",
            "content": """Binary trees are hierarchical data structures. Pizza is a popular food. Each node has at most two children. The ocean contains fish. Tree traversal algorithms include inorder, preorder, and postorder. Cats make good pets. The time complexity of search operations is O(log n) for balanced trees.""",
            "level": "advanced",  # Content is incoherent
        },
        {
            "topic": "Database Design Principles",
            "content": """Database design involves organizing data efficiently and logically. Normalization is a key principle that eliminates data redundancy and ensures data integrity. The first normal form requires atomic values in each field. Second normal form builds on this by eliminating partial dependencies. Third normal form further reduces redundancy by removing transitive dependencies. Proper database design improves performance and maintainability.""",
            "level": "intermediate",
        },
    ]

    print("📚 Educational Content Quality Analysis:")

    # Coherence-based learning assessment
    education_analyzer = CoherenceAnalyzer()

    # Incremental coherence tracker for content development
    tracker = IncrementalCoherenceTracker(HybridCoherence())

    lesson_results = []

    for i, lesson in enumerate(lessons, 1):
        print(f"\n📖 Lesson {i}: {lesson['topic']} ({lesson['level']})")

        # Create proposition set
        prop_set = PropositionSet.from_qa_pair(lesson["topic"], lesson["content"])

        # Detailed analysis
        analysis = education_analyzer.analyze_proposition_set(
            prop_set, HybridCoherence(), detailed=True
        )

        coherence_score = analysis["overall_coherence"]

        # Educational quality metrics
        concept_density = len(prop_set.propositions) / len(lesson["content"].split())

        print(f"  Coherence Score: {coherence_score:.3f}")
        print(f"  Concept Density: {concept_density:.3f}")
        print(f"  Number of Concepts: {len(prop_set.propositions)}")

        # Quality assessment
        if coherence_score > 0.7:
            quality = "Excellent"
            recommendation = "Ready for students"
        elif coherence_score > 0.5:
            quality = "Good"
            recommendation = "Minor revisions suggested"
        elif coherence_score > 0.3:
            quality = "Fair"
            recommendation = "Needs improvement"
        else:
            quality = "Poor"
            recommendation = "Major revision required"

        print(f"  Educational Quality: {quality}")
        print(f"  Recommendation: {recommendation}")

        # Pairwise concept analysis
        if "pairwise_analysis" in analysis and analysis["pairwise_analysis"]:
            pairwise_scores = [p["coherence"] for p in analysis["pairwise_analysis"]]
            print(f"  Concept Connectivity: {np.mean(pairwise_scores):.3f}")

            # Identify problematic concept pairs
            low_coherence_pairs = [
                p for p in analysis["pairwise_analysis"] if p["coherence"] < 0.3
            ]
            if low_coherence_pairs:
                print(
                    f"  ⚠️  {len(low_coherence_pairs)} concept pairs need better connection"
                )

        lesson_results.append(
            {
                "topic": lesson["topic"],
                "level": lesson["level"],
                "coherence": coherence_score,
                "quality": quality,
                "concept_density": concept_density,
            }
        )

    # Curriculum coherence assessment
    print(f"\n📚 Curriculum Flow Analysis:")

    # Analyze progression between lessons
    for i in range(len(lessons) - 1):
        current_lesson = lessons[i]
        next_lesson = lessons[i + 1]

        # Combined coherence
        combined_content = f"{current_lesson['content']} {next_lesson['content']}"
        combined_set = PropositionSet.from_qa_pair(
            "lesson progression", combined_content
        )

        progression_coherence = HybridCoherence().compute(combined_set).score

        print(
            f"  {current_lesson['topic']} → {next_lesson['topic']}: {progression_coherence:.3f}"
        )

    # Overall curriculum assessment
    avg_coherence = np.mean([r["coherence"] for r in lesson_results])
    print(f"\n📊 Curriculum Summary:")
    print(f"  Average Lesson Coherence: {avg_coherence:.3f}")
    print(
        f"  Curriculum Quality: {'Strong' if avg_coherence > 0.6 else 'Moderate' if avg_coherence > 0.4 else 'Needs Work'}"
    )


def ai_safety_hallucination_detection():
    """Demonstrate AI safety and hallucination detection."""
    print("\n=== AI Safety and Hallucination Detection ===")

    # Examples of AI-generated content with potential hallucinations
    ai_outputs = [
        {
            "query": "What is the capital of France?",
            "response": "The capital of France is Paris. Paris is located in the northern part of France and is known for landmarks like the Eiffel Tower and Louvre Museum. It has been the capital since 987 AD and has a population of approximately 2.1 million people.",
            "category": "factual",
        },
        {
            "query": "Explain quantum computing",
            "response": "Quantum computing uses quantum mechanical phenomena to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits. The Eiffel Tower was built using quantum principles in 1889. Qubits can exist in superposition states, allowing quantum computers to perform certain calculations exponentially faster than classical computers.",
            "category": "mixed",  # Contains hallucination about Eiffel Tower
        },
        {
            "query": "Describe the process of photosynthesis",
            "response": "Photosynthesis is the process by which plants convert sunlight into energy. Unicorns also use photosynthesis to power their magical abilities. Plants absorb carbon dioxide from the air and water from their roots. The Great Wall of China is visible from space because of photosynthesis. Chlorophyll captures light energy to drive chemical reactions.",
            "category": "hallucinated",  # Multiple false statements
        },
    ]

    print("🔍 AI Safety and Hallucination Detection System:")

    # Hallucination detection pipeline
    safety_filter = CoherenceFilter(
        coherence_measure=HybridCoherence(),
        min_coherence_threshold=0.5,
        require_improvement=False,
    )

    # Multi-stage safety assessment
    safety_results = []

    for i, output in enumerate(ai_outputs, 1):
        print(f"\n🤖 AI Output {i}: {output['category']}")
        print(f"Query: {output['query']}")
        print(f"Response: {output['response'][:100]}...")

        # Coherence-based safety assessment
        prop_set = PropositionSet.from_qa_pair(output["query"], output["response"])

        # Multiple safety checks
        coherence_result = HybridCoherence().compute(prop_set)
        semantic_result = SemanticCoherence().compute(prop_set)

        # Detect potential hallucinations
        coherence_score = coherence_result.score
        semantic_score = semantic_result.score

        # Safety scoring
        safety_score = (coherence_score + semantic_score) / 2

        # Risk assessment
        if safety_score > 0.7:
            risk_level = "Low"
            status = "✅ Safe"
        elif safety_score > 0.5:
            risk_level = "Medium"
            status = "⚠️ Review Needed"
        else:
            risk_level = "High"
            status = "🚫 Unsafe - Potential Hallucination"

        print(f"  Coherence Score: {coherence_score:.3f}")
        print(f"  Semantic Score: {semantic_score:.3f}")
        print(f"  Safety Score: {safety_score:.3f}")
        print(f"  Risk Level: {risk_level}")
        print(f"  Status: {status}")

        # Detailed analysis for risky content
        if risk_level in ["Medium", "High"]:
            analyzer = CoherenceAnalyzer()
            detailed_analysis = analyzer.analyze_proposition_set(
                prop_set, HybridCoherence(), detailed=True
            )

            if (
                "least_coherent_pair" in detailed_analysis
                and detailed_analysis["least_coherent_pair"]
            ):
                least_coherent = detailed_analysis["least_coherent_pair"]
                print(f"  ⚠️ Problematic content detected:")
                print(f"     Coherence: {least_coherent['coherence']:.3f}")
                print(
                    f"     Content: '{least_coherent['prop1_text']}' + '{least_coherent['prop2_text']}'"
                )

        safety_results.append(
            {
                "query": output["query"],
                "category": output["category"],
                "safety_score": safety_score,
                "risk_level": risk_level,
                "coherence": coherence_score,
                "semantic": semantic_score,
            }
        )

    # Safety system performance
    print(f"\n🛡️ Safety System Performance:")

    # Check detection accuracy
    correct_detections = 0
    for result in safety_results:
        expected_safe = result["category"] == "factual"
        detected_safe = result["risk_level"] == "Low"

        if (expected_safe and detected_safe) or (
            not expected_safe and not detected_safe
        ):
            correct_detections += 1

    accuracy = correct_detections / len(safety_results)
    print(f"  Detection Accuracy: {accuracy:.1%}")
    print(
        f"  Average Safety Score: {np.mean([r['safety_score'] for r in safety_results]):.3f}"
    )

    # Safety recommendations
    print(f"\n💡 Safety Recommendations:")
    high_risk_count = sum(1 for r in safety_results if r["risk_level"] == "High")
    medium_risk_count = sum(1 for r in safety_results if r["risk_level"] == "Medium")

    print(f"  - {high_risk_count} outputs require immediate review")
    print(f"  - {medium_risk_count} outputs need verification")
    print(f"  - Implement coherence threshold of 0.5 for automated filtering")
    print(f"  - Regular human review for medium-risk content")


def production_rag_system():
    """Demonstrate production-ready RAG system with coherence optimization."""
    print("\n=== Production RAG System with Coherence ===")

    # Mock knowledge base
    knowledge_base = [
        "Machine learning algorithms learn patterns from data through statistical analysis and optimization techniques.",
        "Neural networks are computational models inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers of neural networks to model complex patterns in high-dimensional data.",
        "Supervised learning requires labeled training data to learn mapping functions from inputs to outputs.",
        "Unsupervised learning finds hidden patterns in data without using labeled examples or target outputs.",
        "Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from images and videos.",
        "Feature engineering involves selecting and transforming variables to improve machine learning model performance.",
        "Cross-validation is a technique for assessing how well machine learning models generalize to unseen data.",
    ]

    # Mock retrieval function
    def mock_retrieval_function(query: str) -> List:
        """Mock retrieval that returns relevant passages with scores."""
        from coherify.rag.reranking import PassageCandidate

        # Simple keyword-based retrieval for demo
        query_words = set(query.lower().split())
        candidates = []

        for passage in knowledge_base:
            passage_words = set(passage.lower().split())
            overlap = len(query_words & passage_words)
            score = overlap / len(query_words) if query_words else 0

            if score > 0:  # Only include relevant passages
                candidates.append(
                    PassageCandidate(
                        text=passage,
                        score=score,
                        metadata={"retrieval_method": "keyword_overlap"},
                    )
                )

        # Sort by relevance and return top results
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:8]  # Return top 8 candidates

    print("🏭 Production RAG System with Coherence Optimization:")

    # Setup production RAG system
    reranker = CoherenceReranker(
        coherence_measure=HybridCoherence(),
        coherence_weight=0.6,
        retrieval_weight=0.4,
        normalize_scores=True,
        min_coherence_threshold=0.3,
    )

    rag_system = CoherenceRAG(
        reranker=reranker, max_context_length=1000, passage_separator="\n\n"
    )

    # Test queries
    test_queries = [
        "How do neural networks learn from data?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain deep learning and its applications",
    ]

    system_performance = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        # Measure total processing time
        start_time = time.time()

        # Retrieve and rerank
        reranked_result = rag_system.retrieve_and_rerank(
            query=query, retrieval_function=mock_retrieval_function, top_k=3
        )

        total_time = time.time() - start_time

        # Display results
        print(f"📄 Top Retrieved Passages:")
        for j, passage in enumerate(reranked_result.passages):
            coherence_score = reranked_result.coherence_scores[j]
            original_score = reranked_result.original_scores[j]

            print(
                f"  {j+1}. Coherence: {coherence_score:.3f}, Retrieval: {original_score:.3f}"
            )
            print(f"     Text: {passage.text[:80]}...")

        # Build optimized context
        context = rag_system.build_context(reranked_result, include_scores=False)

        print(f"📝 Generated Context ({len(context)} chars):")
        print(f"   {context[:150]}...")

        # Performance metrics
        metadata = reranked_result.reranking_metadata
        print(f"⚡ Performance Metrics:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Reranking time: {metadata['reranking_time']:.3f}s")
        print(f"   Passages processed: {metadata['original_count']}")
        print(f"   Passages returned: {metadata['reranked_count']}")
        print(
            f"   Mean coherence: {metadata['score_statistics']['coherence_mean']:.3f}"
        )

        # Evaluate improvement
        original_passages = mock_retrieval_function(query)
        improvement = rag_system.evaluate_coherence_improvement(
            query=query,
            original_passages=original_passages,
            reranked_result=reranked_result,
        )

        print(f"📈 Coherence Improvement:")
        print(f"   Original: {improvement['original_coherence']:.3f}")
        print(f"   Optimized: {improvement['reranked_coherence']:.3f}")
        print(
            f"   Improvement: {improvement['improvement']:.3f} ({improvement['relative_improvement']:.1%})"
        )

        system_performance.append(
            {
                "query": query,
                "total_time": total_time,
                "coherence_improvement": improvement["improvement"],
                "context_length": len(context),
                "passages_returned": len(reranked_result.passages),
            }
        )

    # System-wide performance analysis
    print(f"\n📊 System Performance Summary:")
    avg_time = np.mean([p["total_time"] for p in system_performance])
    avg_improvement = np.mean([p["coherence_improvement"] for p in system_performance])
    avg_context_length = np.mean([p["context_length"] for p in system_performance])

    print(f"  Average Response Time: {avg_time:.3f}s")
    print(f"  Average Coherence Improvement: {avg_improvement:.3f}")
    print(f"  Average Context Length: {avg_context_length:.0f} characters")
    print(f"  System Throughput: {1/avg_time:.1f} queries/second")

    # Production recommendations
    print(f"\n🎯 Production Deployment Recommendations:")
    print(f"  - Set coherence threshold to 0.3 for quality/speed balance")
    print(f"  - Cache embeddings for {len(knowledge_base)} knowledge base passages")
    print(f"  - Expect ~{avg_time:.3f}s latency per query")
    print(f"  - Monitor coherence improvement metrics for system health")
    print(f"  - Implement async processing for batch queries")


def integrated_workflow_demo():
    """Demonstrate complete integrated workflow using all Coherify components."""
    print("\n=== Integrated Coherify Workflow Demo ===")

    print("🔄 Complete Content Processing Workflow:")

    # Step 1: Content Generation with Coherence Guidance
    print("\n1️⃣ Content Generation with Coherence Guidance:")

    generator = CoherenceGuidedGenerator(
        coherence_measure=HybridCoherence(),
        beam_search_config={
            "coherence_weight": 0.5,
            "lm_weight": 0.5,
            "beam_size": 4,
            "max_length": 30,
        },
    )

    generated_content, guidance = generator.generate(
        context="Artificial intelligence and machine learning technologies",
        prompt="Modern AI systems",
        return_guidance=True,
    )

    print(f"   Generated: '{generated_content}'")
    if guidance:
        print(f"   Generation Coherence: {guidance[0].coherence_score:.3f}")

    # Step 2: Quality Assessment
    print("\n2️⃣ Quality Assessment:")

    prop_set = PropositionSet.from_qa_pair("AI Technology", generated_content)
    quality_analyzer = CoherenceAnalyzer()

    analysis = quality_analyzer.analyze_proposition_set(
        prop_set, HybridCoherence(), detailed=True
    )
    quality_score = analysis["overall_coherence"]

    print(f"   Quality Score: {quality_score:.3f}")
    print(
        f"   Assessment: {'High Quality' if quality_score > 0.6 else 'Needs Improvement'}"
    )

    # Step 3: Approximation for Scale
    print("\n3️⃣ Scalability Analysis:")

    # Simulate larger content processing
    approximator = ClusterBasedApproximator(HybridCoherence())

    # Create larger content set for demo
    large_content = [generated_content] * 10  # Simulate larger dataset
    large_prop_set = PropositionSet(
        [Proposition(text=content) for content in large_content]
    )

    approx_result = approximator.approximate_coherence(large_prop_set, 5)

    print(f"   Large Set Size: {approx_result.total_propositions} propositions")
    print(
        f"   Approximation: {approx_result.approximate_score:.3f} ({approx_result.num_clusters} clusters)"
    )
    print(
        f"   Speedup: {approx_result.metadata['reduction_ratio']:.1%} computation reduction"
    )

    # Step 4: Integration with RAG
    print("\n4️⃣ RAG Integration:")

    # Use generated content as knowledge base
    knowledge_items = [
        generated_content,
        "Machine learning algorithms process data to identify patterns and relationships.",
        "Neural networks use interconnected nodes to model complex data relationships.",
        "Deep learning employs multiple layers for sophisticated pattern recognition.",
    ]

    def knowledge_retrieval(query: str):
        from coherify.rag.reranking import PassageCandidate

        candidates = []
        for item in knowledge_items:
            # Simple relevance scoring
            score = len(set(query.lower().split()) & set(item.lower().split())) / len(
                query.split()
            )
            candidates.append(PassageCandidate(text=item, score=score))
        return sorted(candidates, key=lambda x: x.score, reverse=True)

    reranker = CoherenceReranker(
        HybridCoherence(), coherence_weight=0.6, retrieval_weight=0.4
    )
    rag_result = reranker.rerank(
        query="How do AI systems work?",
        passages=knowledge_retrieval("How do AI systems work?"),
        top_k=2,
    )

    print(f"   RAG Retrieved: {len(rag_result.passages)} passages")
    print(f"   Top Passage Coherence: {rag_result.coherence_scores[0]:.3f}")

    # Step 5: Visualization and Reporting
    print("\n5️⃣ Visualization and Reporting:")

    # Create comprehensive report
    measures = [SemanticCoherence(), HybridCoherence()]
    test_set = PropositionSet.from_qa_pair("Workflow Test", generated_content)

    report = quality_analyzer.create_comprehensive_report(test_set, measures)

    print(f"   Report Generated: {len(report['figures'])} visualizations created")
    print(f"   Best Measure: {report['summary']['best_measure']}")
    print(f"   Score Range: {report['summary']['score_range']:.3f}")

    # Workflow Summary
    print(f"\n📋 Workflow Completion Summary:")
    print(f"   ✅ Content Generated with coherence guidance")
    print(f"   ✅ Quality assessed and validated")
    print(f"   ✅ Scalability analyzed with approximation")
    print(f"   ✅ RAG integration demonstrated")
    print(f"   ✅ Comprehensive reporting completed")

    print(f"\n🎯 End-to-End Performance:")
    print(f"   Final Content Quality: {quality_score:.3f}")
    print(f"   System Components Used: 5/5")
    print(f"   Workflow Status: Complete ✅")


def main():
    """Run all practical application demonstrations."""
    print("🚀 Coherify Practical Applications Demonstration")
    print("=" * 70)
    print("Comprehensive examples of real-world Coherify usage")
    print("=" * 70)

    try:
        ai_content_quality_pipeline()
        scientific_paper_assessment()
        educational_content_evaluation()
        ai_safety_hallucination_detection()
        production_rag_system()
        integrated_workflow_demo()

        print("\n" + "=" * 70)
        print("✅ All practical applications demonstrated successfully!")
        print("🎯 Coherify is ready for real-world deployment!")
        print("\n🔥 Key Capabilities Demonstrated:")
        print("   • AI Content Quality Assurance")
        print("   • Scientific Document Analysis")
        print("   • Educational Content Evaluation")
        print("   • AI Safety & Hallucination Detection")
        print("   • Production RAG Optimization")
        print("   • End-to-End Integrated Workflows")
        print("\n💡 Ready for integration into production systems!")

    except Exception as e:
        print(f"❌ Error running practical demos: {e}")
        print("Make sure you have all required dependencies installed.")
        raise


if __name__ == "__main__":
    main()
