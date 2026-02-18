#!/usr/bin/env python3
"""
USF BIOS - End-to-End Test Dataset Preparation
Downloads real datasets from HuggingFace and converts them to USF BIOS formats.
Creates ~2000 samples per dataset type for realistic training speed testing.

Dataset types created:
1. SFT (alpaca format) - instruction/input/output
2. SFT (messages format) - messages with role/content
3. RLHF Offline Preference - prompt/chosen/rejected (for DPO, ORPO, SimPO, CPO)
4. KTO - messages with label (for KTO)
5. RLHF Online Prompts - prompt only (for PPO, GRPO)
6. Pre-training - raw text (for PT/CPT)
"""

import json
import os
import sys
import random
import argparse
from pathlib import Path

# Target directory for test datasets
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TARGET_SAMPLES = 2000


def download_sft_dataset(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Download and convert an SFT dataset from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"[1/6] Preparing SFT Dataset ({num_samples} samples)")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        # Use yahma/alpaca-cleaned - high quality SFT dataset
        print("  Downloading yahma/alpaca-cleaned from HuggingFace...")
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        
        # Shuffle and take num_samples
        ds = ds.shuffle(seed=42)
        samples = []
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"]
            })
        
        output_path = os.path.join(output_dir, "sft_alpaca_2k.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved {len(samples)} SFT samples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to download SFT dataset: {e}")
        print("  → Generating synthetic SFT dataset instead...")
        return generate_synthetic_sft(output_dir, num_samples)


def generate_synthetic_sft(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Generate synthetic SFT data as fallback."""
    topics = [
        ("Explain what {topic} is and why it matters.", ""),
        ("Write a Python function to {topic}.", ""),
        ("Compare and contrast {topic}.", ""),
        ("List the key benefits of {topic}.", ""),
        ("How would you implement {topic} in a production system?", ""),
        ("What are common mistakes when dealing with {topic}?", ""),
        ("Describe the architecture of {topic}.", ""),
        ("Explain {topic} to a beginner.", ""),
    ]
    
    subjects = [
        "machine learning", "neural networks", "data structures", "algorithms",
        "databases", "REST APIs", "microservices", "containerization",
        "cloud computing", "DevOps", "CI/CD pipelines", "unit testing",
        "code review", "agile methodology", "version control", "debugging",
        "performance optimization", "security best practices", "API design",
        "distributed systems", "caching strategies", "load balancing",
        "message queues", "event-driven architecture", "serverless computing",
        "GraphQL", "WebSocket", "OAuth 2.0", "JWT tokens", "encryption",
        "hashing algorithms", "sorting algorithms", "graph algorithms",
        "dynamic programming", "recursion", "object-oriented programming",
        "functional programming", "design patterns", "SOLID principles",
        "clean code", "refactoring", "technical debt", "monitoring",
        "logging", "error handling", "input validation", "rate limiting",
        "pagination", "search optimization", "indexing strategies",
        "data modeling", "normalization", "denormalization",
    ]
    
    samples = []
    for i in range(num_samples):
        template, inp = random.choice(topics)
        subject = random.choice(subjects)
        instruction = template.format(topic=subject)
        output = f"Here is a detailed explanation about {subject}:\n\n"
        output += f"{subject.title()} is a fundamental concept in software engineering. "
        output += f"It involves understanding the principles and practices that make systems reliable, "
        output += f"scalable, and maintainable. Key aspects include:\n\n"
        output += f"1. **Core Concepts**: Understanding the theoretical foundations\n"
        output += f"2. **Practical Application**: Implementing in real-world scenarios\n"
        output += f"3. **Best Practices**: Following industry standards\n"
        output += f"4. **Common Pitfalls**: Avoiding frequent mistakes\n\n"
        output += f"This knowledge is essential for building robust software systems."
        
        samples.append({
            "instruction": instruction,
            "input": inp,
            "output": output
        })
    
    output_path = os.path.join(output_dir, "sft_alpaca_2k.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Generated {len(samples)} synthetic SFT samples to {output_path}")
    return output_path


def download_rlhf_preference_dataset(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Download and convert RLHF preference dataset (prompt/chosen/rejected)."""
    print(f"\n{'='*60}")
    print(f"[2/6] Preparing RLHF Preference Dataset ({num_samples} samples)")
    print(f"      For: DPO, ORPO, SimPO, CPO")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        # Use Intel/orca_dpo_pairs - good quality DPO dataset  
        print("  Downloading Intel/orca_dpo_pairs from HuggingFace...")
        ds = load_dataset("Intel/orca_dpo_pairs", split="train")
        
        ds = ds.shuffle(seed=42)
        samples = []
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            # Convert to USF BIOS format: prompt/chosen/rejected
            prompt = row.get("question", row.get("prompt", ""))
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            
            if prompt and chosen and rejected:
                samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        
        output_path = os.path.join(output_dir, "rlhf_preference_2k.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved {len(samples)} RLHF preference samples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to download preference dataset: {e}")
        print("  → Generating synthetic preference dataset instead...")
        return generate_synthetic_preference(output_dir, num_samples)


def generate_synthetic_preference(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Generate synthetic preference data as fallback."""
    prompts = [
        "Explain how {topic} works.",
        "What is the best approach to {topic}?",
        "How do you implement {topic}?",
        "Describe the advantages of {topic}.",
        "What are the key considerations for {topic}?",
    ]
    
    subjects = [
        "binary search", "hash tables", "linked lists", "tree traversal",
        "graph algorithms", "dynamic programming", "recursion", "sorting",
        "caching", "load balancing", "database indexing", "API design",
        "error handling", "input validation", "authentication", "authorization",
        "encryption", "compression", "serialization", "deserialization",
        "multithreading", "async programming", "event loops", "memory management",
        "garbage collection", "virtual memory", "file systems", "networking",
        "HTTP protocol", "TCP/IP", "DNS resolution", "SSL/TLS",
        "containerization", "orchestration", "CI/CD", "monitoring",
        "logging", "alerting", "tracing", "profiling",
    ]
    
    samples = []
    for i in range(num_samples):
        template = random.choice(prompts)
        subject = random.choice(subjects)
        prompt = template.format(topic=subject)
        
        chosen = f"Here is a comprehensive explanation of {subject}:\n\n"
        chosen += f"{subject.title()} is an important concept. It involves several key aspects:\n\n"
        chosen += f"1. **Understanding the fundamentals**: The core principle behind {subject} is efficiency and reliability.\n"
        chosen += f"2. **Implementation details**: When implementing {subject}, consider edge cases, performance, and maintainability.\n"
        chosen += f"3. **Best practices**: Always test thoroughly, document your approach, and consider scalability.\n"
        chosen += f"4. **Common patterns**: Standard patterns include modular design and separation of concerns.\n\n"
        chosen += f"This approach ensures robust and maintainable solutions."
        
        rejected = f"{subject} is simple. Just do it the obvious way and it should work fine."
        
        samples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    output_path = os.path.join(output_dir, "rlhf_preference_2k.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Generated {len(samples)} synthetic preference samples to {output_path}")
    return output_path


def download_kto_dataset(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Create KTO dataset (messages + label format)."""
    print(f"\n{'='*60}")
    print(f"[3/6] Preparing KTO Dataset ({num_samples} samples)")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        # Use argilla/ultrafeedback-binarized-preferences for KTO
        print("  Downloading argilla/ultrafeedback-binarized-preferences...")
        ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        
        ds = ds.shuffle(seed=42)
        samples = []
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])
            
            if prompt and chosen:
                # Good example (label=true)
                chosen_text = chosen[-1]["content"] if isinstance(chosen, list) and len(chosen) > 0 else str(chosen)
                samples.append({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen_text}
                    ],
                    "label": True
                })
            
            if prompt and rejected and len(samples) < num_samples:
                # Bad example (label=false)
                rejected_text = rejected[-1]["content"] if isinstance(rejected, list) and len(rejected) > 0 else str(rejected)
                samples.append({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected_text}
                    ],
                    "label": False
                })
        
        # Shuffle to mix true/false labels
        random.shuffle(samples)
        samples = samples[:num_samples]
        
        output_path = os.path.join(output_dir, "kto_messages_2k.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved {len(samples)} KTO samples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to download KTO dataset: {e}")
        print("  → Generating synthetic KTO dataset instead...")
        return generate_synthetic_kto(output_dir, num_samples)


def generate_synthetic_kto(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Generate synthetic KTO data as fallback."""
    questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "Explain the concept of overfitting.",
        "What is gradient descent?",
        "How do transformers work?",
        "What is regularization?",
        "Explain batch normalization.",
        "What is transfer learning?",
        "How does backpropagation work?",
        "What is the attention mechanism?",
        "Explain convolutional neural networks.",
        "What is reinforcement learning?",
        "How do GANs work?",
        "What is a loss function?",
        "Explain the bias-variance tradeoff.",
        "What is feature engineering?",
        "How does cross-validation work?",
        "What is ensemble learning?",
        "Explain the vanishing gradient problem.",
        "What is data augmentation?",
    ]
    
    samples = []
    for i in range(num_samples):
        q = random.choice(questions)
        is_good = random.random() > 0.5
        
        if is_good:
            answer = f"This is a well-explained answer about {q.lower().replace('?', '')}. "
            answer += "It covers the fundamental concepts, practical applications, and key considerations. "
            answer += "The explanation includes relevant examples and best practices that help understand the topic thoroughly."
            label = True
        else:
            answer = f"I don't really know about that. Maybe look it up online."
            label = False
        
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer}
            ],
            "label": label
        })
    
    output_path = os.path.join(output_dir, "kto_messages_2k.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Generated {len(samples)} synthetic KTO samples to {output_path}")
    return output_path


def download_online_prompts_dataset(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Create online RL prompts dataset (prompt only, for PPO/GRPO)."""
    print(f"\n{'='*60}")
    print(f"[4/6] Preparing Online RL Prompts Dataset ({num_samples} samples)")
    print(f"      For: PPO, GRPO")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        # Use OpenAssistant prompts
        print("  Downloading HuggingFaceH4/ultrafeedback_binarized prompts...")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        
        ds = ds.shuffle(seed=42)
        samples = []
        seen_prompts = set()
        for row in ds:
            if len(samples) >= num_samples:
                break
            prompt = row.get("prompt", "")
            if prompt and prompt not in seen_prompts:
                seen_prompts.add(prompt)
                samples.append({"prompt": prompt})
        
        output_path = os.path.join(output_dir, "online_prompts_2k.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved {len(samples)} online prompt samples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to download prompts dataset: {e}")
        print("  → Generating synthetic prompts dataset instead...")
        return generate_synthetic_prompts(output_dir, num_samples)


def generate_synthetic_prompts(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Generate synthetic prompts for online RL."""
    templates = [
        "Explain {topic} in detail.",
        "Write a Python implementation of {topic}.",
        "Compare {topic} with its alternatives.",
        "What are the best practices for {topic}?",
        "How would you optimize {topic}?",
        "Describe the architecture of {topic}.",
        "What are common mistakes when implementing {topic}?",
        "How does {topic} work under the hood?",
        "What is the time complexity of {topic}?",
        "When should you use {topic}?",
    ]
    
    subjects = [
        "binary search", "merge sort", "quick sort", "hash tables",
        "linked lists", "binary trees", "graph traversal", "BFS",
        "DFS", "dynamic programming", "greedy algorithms", "backtracking",
        "A* search", "Dijkstra's algorithm", "union-find", "tries",
        "segment trees", "fenwick trees", "red-black trees", "B-trees",
        "skip lists", "bloom filters", "LRU cache", "consistent hashing",
        "MapReduce", "Spark", "Kafka", "Redis",
        "PostgreSQL", "MongoDB", "Elasticsearch", "gRPC",
        "REST APIs", "WebSockets", "OAuth", "JWT",
        "Docker", "Kubernetes", "Terraform", "Ansible",
        "React", "Vue.js", "Node.js", "FastAPI",
    ]
    
    samples = []
    for i in range(num_samples):
        template = random.choice(templates)
        subject = random.choice(subjects)
        samples.append({"prompt": template.format(topic=subject)})
    
    output_path = os.path.join(output_dir, "online_prompts_2k.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Generated {len(samples)} synthetic prompt samples to {output_path}")
    return output_path


def download_pretraining_dataset(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Download pre-training dataset (raw text format)."""
    print(f"\n{'='*60}")
    print(f"[5/6] Preparing Pre-training Dataset ({num_samples} samples)")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        # Use wikitext for pretraining
        print("  Downloading wikitext-2-raw-v1 from HuggingFace...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        samples = []
        for row in ds:
            if len(samples) >= num_samples:
                break
            text = row.get("text", "").strip()
            # Only include non-empty, substantial text (> 50 chars)
            if text and len(text) > 50 and not text.startswith("="):
                samples.append({"text": text})
        
        # If not enough samples, duplicate with slight variations
        while len(samples) < num_samples:
            original = random.choice(samples[:min(len(samples), 500)])
            samples.append(original)
        
        samples = samples[:num_samples]
        
        output_path = os.path.join(output_dir, "pt_raw_text_2k.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved {len(samples)} pre-training samples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to download pre-training dataset: {e}")
        print("  → Generating synthetic pre-training dataset instead...")
        return generate_synthetic_pretraining(output_dir, num_samples)


def generate_synthetic_pretraining(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Generate synthetic pre-training data as fallback."""
    topics = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning.",
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's extensive standard library and large ecosystem of third-party packages make it suitable for a wide range of applications.",
        "Database management systems are software applications that interact with users, other applications, and the database itself to capture and analyze data. Common types include relational databases like PostgreSQL and MySQL, and NoSQL databases like MongoDB and Cassandra.",
        "Cloud computing provides on-demand access to computing resources including servers, storage, databases, networking, software, and analytics over the internet. Major providers include Amazon Web Services, Microsoft Azure, and Google Cloud Platform.",
        "Cybersecurity involves protecting computer systems, networks, and data from digital attacks, unauthorized access, and data breaches. Key areas include network security, application security, information security, and operational security.",
        "Artificial neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes organized in layers that process information using connectionist approaches to computation. Deep learning uses neural networks with many layers.",
        "Software engineering is the systematic application of engineering approaches to the development of software. It encompasses requirements analysis, design, implementation, testing, deployment, and maintenance of software systems.",
        "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data. It employs techniques from many fields including machine learning, statistical analysis, and data visualization.",
    ]
    
    samples = []
    for i in range(num_samples):
        base = random.choice(topics)
        # Add some variation
        prefix_words = ["Furthermore, ", "Additionally, ", "Moreover, ", "In practice, ", "Specifically, ", ""]
        suffix = f" This is particularly relevant in modern {random.choice(['software development', 'data science', 'engineering', 'technology', 'research'])} contexts."
        text = random.choice(prefix_words) + base + suffix
        samples.append({"text": text})
    
    output_path = os.path.join(output_dir, "pt_raw_text_2k.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Generated {len(samples)} synthetic pre-training samples to {output_path}")
    return output_path


def download_sft_messages_dataset(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Create SFT messages format dataset (multi-turn conversations)."""
    print(f"\n{'='*60}")
    print(f"[6/6] Preparing SFT Messages Dataset ({num_samples} samples)")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        # Use HuggingFaceH4/no_robots for SFT messages format
        print("  Downloading HuggingFaceH4/no_robots from HuggingFace...")
        ds = load_dataset("HuggingFaceH4/no_robots", split="train")
        
        ds = ds.shuffle(seed=42)
        samples = []
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            messages = row.get("messages", [])
            if messages and len(messages) >= 2:
                samples.append({"messages": messages})
        
        # If not enough, pad with synthetic
        while len(samples) < num_samples:
            q = f"Explain topic number {len(samples)} in detail."
            a = f"Here is a detailed explanation of topic {len(samples)}. It covers the fundamentals, practical applications, and best practices."
            samples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })
        
        samples = samples[:num_samples]
        
        output_path = os.path.join(output_dir, "sft_messages_2k.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        
        print(f"  ✓ Saved {len(samples)} SFT messages samples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Failed to download SFT messages dataset: {e}")
        print("  → Generating synthetic SFT messages dataset instead...")
        return generate_synthetic_sft_messages(output_dir, num_samples)


def generate_synthetic_sft_messages(output_dir: str, num_samples: int = TARGET_SAMPLES):
    """Generate synthetic SFT messages data."""
    questions = [
        "What is {topic}?",
        "How does {topic} work?",
        "Explain {topic} with examples.",
        "What are the benefits of {topic}?",
        "Compare {topic} with alternatives.",
    ]
    
    subjects = [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "reinforcement learning", "transfer learning",
        "data augmentation", "model compression", "quantization",
        "knowledge distillation", "federated learning", "meta-learning",
        "few-shot learning", "zero-shot learning", "self-supervised learning",
        "contrastive learning", "generative models", "diffusion models",
        "attention mechanisms", "transformer architecture",
    ]
    
    samples = []
    for i in range(num_samples):
        template = random.choice(questions)
        subject = random.choice(subjects)
        q = template.format(topic=subject)
        a = f"{subject.title()} is an important concept in modern AI. "
        a += f"It involves understanding and applying specific techniques to solve complex problems. "
        a += f"Key aspects include theoretical foundations, practical implementations, and real-world applications. "
        a += f"When working with {subject}, it's important to consider performance, scalability, and reliability."
        
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    output_path = os.path.join(output_dir, "sft_messages_2k.jsonl")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  ✓ Generated {len(samples)} synthetic SFT messages samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare test datasets for USF BIOS E2E testing")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for datasets")
    parser.add_argument("--num-samples", type=int, default=TARGET_SAMPLES, help="Number of samples per dataset")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    num_samples = args.num_samples
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("USF BIOS - Test Dataset Preparation")
    print(f"Output directory: {output_dir}")
    print(f"Samples per dataset: {num_samples}")
    print("=" * 60)
    
    datasets = {}
    
    # 1. SFT (alpaca format)
    datasets["sft_alpaca"] = download_sft_dataset(output_dir, num_samples)
    
    # 2. RLHF Offline Preference (prompt/chosen/rejected)
    datasets["rlhf_preference"] = download_rlhf_preference_dataset(output_dir, num_samples)
    
    # 3. KTO (messages + label)
    datasets["kto"] = download_kto_dataset(output_dir, num_samples)
    
    # 4. Online RL Prompts (prompt only)
    datasets["online_prompts"] = download_online_prompts_dataset(output_dir, num_samples)
    
    # 5. Pre-training (raw text)
    datasets["pretraining"] = download_pretraining_dataset(output_dir, num_samples)
    
    # 6. SFT Messages format
    datasets["sft_messages"] = download_sft_messages_dataset(output_dir, num_samples)
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "created_at": str(__import__("datetime").datetime.now()),
        "num_samples_per_dataset": num_samples,
        "datasets": {}
    }
    for name, path in datasets.items():
        if path and os.path.exists(path):
            line_count = sum(1 for _ in open(path))
            file_size = os.path.getsize(path)
            manifest["datasets"][name] = {
                "path": os.path.abspath(path),
                "filename": os.path.basename(path),
                "lines": line_count,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Dataset Preparation Complete!")
    print(f"{'='*60}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"\nDatasets created:")
    for name, info in manifest["datasets"].items():
        print(f"  • {name}: {info['lines']} samples ({info['size_mb']} MB) → {info['filename']}")
    
    return datasets


if __name__ == "__main__":
    main()
