#!/usr/bin/env python3
"""
Generate CSV data with sentences and matching score arrays.
Ensures word count matches array length.
"""

import csv
import random
import numpy as np

# Sample sentences with varying word counts (10-20 words)
sample_sentences = [
    "The quick brown fox jumps over the lazy dog in the park",
    "Machine learning models require large datasets for training effectively",
    "Natural language processing helps computers understand human communication",
    "Artificial intelligence revolutionizes how we interact with technology daily",
    "Deep neural networks can process complex patterns in massive datasets",
    "Computer vision enables machines to recognize objects in images automatically",
    "Reinforcement learning allows agents to learn through trial and error interactions",
    "Transfer learning helps models adapt knowledge across different problem domains",
    "Gradient descent optimizes model parameters by minimizing loss functions iteratively",
    "Convolutional networks excel at extracting features from spatial image data",
    "Recurrent architectures process sequential information using memory mechanisms",
    "Attention mechanisms allow models to focus on relevant parts of input data",
    "Transformer models revolutionized natural language understanding through self attention",
    "Batch normalization stabilizes training by normalizing activations across mini batches",
    "Dropout regularization prevents overfitting by randomly deactivating neurons during training",
    "Supervised learning uses labeled examples to train predictive models on new data",
    "Unsupervised learning discovers hidden patterns in data without explicit labels",
    "Semi supervised approaches combine labeled and unlabeled data for better performance",
    "Feature engineering transforms raw data into meaningful representations for algorithms",
    "Cross validation helps estimate model performance on unseen test data accurately",
    "Hyperparameter tuning optimizes model configuration through systematic search methods",
    "Overfitting occurs when models memorize training data instead of learning general patterns",
    "Regularization techniques prevent overfitting by adding constraints to model complexity",
    "Ensemble methods combine multiple models to achieve better predictive performance together",
    "Bias variance tradeoff balances model complexity with generalization ability carefully",
    "Precision recall metrics evaluate classification model performance on imbalanced datasets",
    "Confusion matrices provide detailed insights into classification error patterns",
    "ROC curves visualize classification performance across different decision thresholds",
    "Loss functions measure prediction errors during model training optimization process",
    "Backpropagation computes gradients efficiently for neural network parameter updates",
    "Activation functions introduce nonlinearity into neural network computational graphs",
    "Pooling layers reduce spatial dimensions in convolutional neural network architectures",
    "Fully connected layers combine features from previous layers in neural networks",
    "Embedding layers map discrete tokens to continuous vector representations efficiently",
    "Tokenization splits text into smaller units for natural language processing tasks",
    "Word embeddings capture semantic relationships between words in vector space",
    "Language models predict next words based on previous context in sequences",
    "Named entity recognition identifies people organizations and locations in text",
    "Sentiment analysis determines emotional tone and opinions expressed in text",
    "Machine translation converts text from one language to another automatically",
    "Question answering systems extract answers from documents based on queries",
    "Text summarization creates concise summaries from longer documents automatically",
    "Chatbots use natural language understanding to converse with users effectively",
    "Speech recognition converts spoken audio into written text transcripts",
    "Image classification assigns category labels to pictures using deep learning",
]

def generate_scores(word_count):
    """Generate random scores array matching word count."""
    # Generate random scores between -0.5 and 0.5
    scores = [round(random.uniform(-0.5, 0.5), 3) for _ in range(word_count)]
    return scores

def array_to_string(arr):
    """Convert array to CSV-friendly string format."""
    return '[' + ','.join(map(str, arr)) + ']'

# Generate 45 rows of data
rows = []
for i in range(45):
    # Use sentence from sample, cycling if needed
    sentence = sample_sentences[i % len(sample_sentences)]
    word_count = len(sentence.split())
    
    # Generate matching score arrays
    plex_scores = generate_scores(word_count)
    lime_scores = generate_scores(word_count)
    
    row = {
        'id': i + 1,
        'sentence': sentence,
        'Plex score': array_to_string(plex_scores),
        'LIME score': array_to_string(lime_scores)
    }
    rows.append(row)
    
    # Verify length match
    print(f"ID {i+1}: {word_count} words, Plex: {len(plex_scores)} scores, LIME: {len(lime_scores)} scores")

# Write to CSV
with open('data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'sentence', 'Plex score', 'LIME score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"\nGenerated {len(rows)} rows in data.csv")
print("Verification: All arrays should match word counts!")

