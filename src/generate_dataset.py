import pandas as pd
import random

# Define syllabus topics and corresponding teaching texts
SYLLABUS_DATA = {
    "Introduction to Python": [
        "Python is a high-level, interpreted programming language.",
        "It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
        "We will cover basic syntax, data types, and control structures."
    ],
    "Data Structures in Python": [
        "This module introduces fundamental data structures.",
        "We will explore lists, tuples, dictionaries, and sets.",
        "Understanding their properties and use cases is crucial for efficient programming."
    ],
    "Object-Oriented Programming (OOP)": [
        "OOP is a programming paradigm based on the concept of 'objects'.",
        "We will learn about classes, objects, inheritance, and polymorphism.",
        "This section will include practical examples of implementing OOP principles in Python."
    ],
    "Introduction to Machine Learning": [
        "Machine learning is a field of study in artificial intelligence.",
        "This course covers the basic concepts of supervised, unsupervised, and reinforcement learning.",
        "We will implement simple algorithms like linear regression."
    ],
    "Natural Language Processing (NLP)": [
        "NLP is a subfield of linguistics, computer science, and artificial intelligence.",
        "It focuses on the interaction between computers and human language.",
        "Topics include tokenization, sentiment analysis, and text classification."
    ]
}

def generate_sentence_pairs(num_samples):
    """Generates positive and negative sentence pairs."""
    pairs = []
    labels = []

    topics = list(SYLLABUS_DATA.keys())

    for _ in range(num_samples):
        # Create a positive pair
        if random.random() > 0.5:
            topic = random.choice(topics)
            syllabus = f"Syllabus: {topic}"
            text = " ".join(random.sample(SYLLABUS_DATA[topic], k=random.randint(2,3)))
            pairs.append([syllabus, f"Text: {text}"])
            labels.append(1)
        # Create a negative pair
        else:
            topic1, topic2 = random.sample(topics, 2)
            syllabus = f"Syllabus: {topic1}"
            text = " ".join(random.sample(SYLLABUS_DATA[topic2], k=random.randint(2,3)))
            pairs.append([syllabus, f"Text: {text}"])
            labels.append(0)

    return pairs, labels

def main():
    """Main function to generate and save datasets."""
    # Generate training data
    train_pairs, train_labels = generate_sentence_pairs(1000)
    train_df = pd.DataFrame(train_pairs, columns=['sentence1', 'sentence2'])
    train_df['label'] = train_labels
    train_df.to_csv('data/train.csv', index=False)

    # Generate test data
    test_pairs, test_labels = generate_sentence_pairs(200)
    test_df = pd.DataFrame(test_pairs, columns=['sentence1', 'sentence2'])
    test_df['label'] = test_labels
    test_df.to_csv('data/test.csv', index=False)

    print("Datasets generated successfully in 'data/' directory.")

if __name__ == "__main__":
    main()
