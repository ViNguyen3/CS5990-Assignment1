# -------------------------------------------------------------------------
# AUTHOR: Vi Nguyen
# FILENAME: final_hw.py 
# SPECIFICATION: For question 8 in the hw1 to find 2 most similar documents using cosine similarity 
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH
# AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays
# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

def read_documents(): 
    documents = []
    #reading the documents in a csv file
    with open('cleaned_documents.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                documents.append (row)
                # print(row)
    return documents

def build_document_term_matrix(document): 
    #Building the document-term matrix by using binary encoding.
    #You must identify each distinct word in the collection without applying any
    # transformations, using
    # the spaces as your character delimiter.
    #--> add your Python code here
    docTermMatrix = []
    distinct_words = set()

    #Splitting by spaces and add to the set 
    for doc in document: 
        words = doc[1].split()
        distinct_words.update(words)

    word_index = {word: i for i, word in enumerate(distinct_words)}

    for doc in document: 
        vector = [0] * len(distinct_words)
        words = doc[1].split()
        for word in words: 
            if word in word_index: 
                vector[word_index[word]] += 1 
        docTermMatrix.append(vector)

     # Write term matrix to file so I can check it 
    with open('term_matrix_output.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Document"] + list(distinct_words))  # Header row with words
        for i, row in enumerate(docTermMatrix):
            writer.writerow([f"Doc {i+1}"] + row)


    return docTermMatrix

def calculate_cosine_similar(vec1, vec2): 
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    lenght_vec1 = pow(sum(pow(v, 2) for v in vec1),0.5) 
    lenght_vec2 = pow(sum(pow(v, 2) for v in vec2),0.5) 
    if lenght_vec1 == 0 or lenght_vec2 == 0: #return 0 to avoid division of 0 
        return 0
    return dot_product/(lenght_vec1 * lenght_vec2)

def find_most_similar_document(docTermMatrix): 
    max_sim = -1 
    most_sim_doc = (-1, -1)

    # Compare the pairwise cosine similarities and store the highest one
    # Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
    # --> Add your Python code here
    for i in range(len(docTermMatrix)):
        for j in range(i + 1, len(docTermMatrix)): 
            similarity = calculate_cosine_similar(docTermMatrix[i], docTermMatrix[j])
            if similarity > max_sim: 
                max_sim = similarity
                most_sim_doc = (i+1, j+1)
    
    return most_sim_doc, max_sim


    

# The most similar documents are document 10 and document 100 with cosine
# similarity = x
# --> Add your Python code here


documents = read_documents()
term_matrix = build_document_term_matrix(documents)
most_similar_docs, max_similarity = find_most_similar_document(term_matrix)
print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity:.4f}.")
