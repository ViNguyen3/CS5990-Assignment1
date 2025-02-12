import csv
import math

def read_documents(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            documents.append(row[0].strip())  # Strip spaces to avoid inconsistencies
    return documents

def build_term_matrix(documents):
    unique_words = set()
    
    for doc in documents:
        words = doc.split()  # Split by spaces
        unique_words.update(words)
    
    unique_words = sorted(unique_words)  # Keep a consistent order
    word_index = {word: i for i, word in enumerate(unique_words)}
    
    term_matrix = []
    for doc in documents:
        vector = [0] * len(unique_words)
        words = doc.split()
        for word in words:
            if word in word_index:
                vector[word_index[word]] = 1  # Binary encoding
        term_matrix.append(vector)
    
    return term_matrix

def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
    magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Avoid division by zero
    return dot_product / (magnitude1 * magnitude2)

def find_most_similar_documents(term_matrix):
    max_similarity = -1
    most_similar_docs = (-1, -1)
    
    for i in range(len(term_matrix)):
        for j in range(i + 1, len(term_matrix)):
            similarity = cosine_similarity(term_matrix[i], term_matrix[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_docs = (i + 1, j + 1)  # Document index starts at 1
    
    return most_similar_docs, max_similarity

# Main execution
documents = read_documents('cleaned_documents.csv')
if not documents:
    print("Error: No documents found in the file.")
else:
    term_matrix = build_term_matrix(documents)
    if not term_matrix or len(term_matrix) < 2:
        print("Error: Not enough documents for comparison.")
    else:
        most_similar_docs, max_similarity = find_most_similar_documents(term_matrix)
        print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity:.4f}.")
