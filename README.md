
# Word2Vec-Based Word Embedding and Similarity Analysis
## NAME: SANJUSHRI A
## REGNO:212223040187
## AIM:
To implement word embedding using the Word2Vec model from Gensim, which captures semantic relationships between words by representing them in a dense vector space.

## THEORY:
Word Embedding refers to the process of converting words into numerical vector representations that preserve semantic relationships. Word2Vec, developed by Google, is one such model that uses neural networks to learn word representations from large corpora.

**Word2Vec has two architectures:**
**CBOW (Continuous Bag of Words):** Predicts a word based on its context.

**Skip-Gram:** Predicts the context based on a word (used in this implementation).

**Key Concepts:**
Gensim Library: A Python library used for unsupervised learning of word embeddings.

Dense Vector Space: Each word is mapped to a high-dimensional vector.

Semantic Similarity: Words that appear in similar contexts have similar vectors.

## PROCEDURE:
STEP . 1 : Import Required Libraries: Use gensim, nltk, and other NLP tools.

STEP . 2 : Prepare Sample Corpus: A set of meaningful sentences is defined.

STEP . 3 : Text Preprocessing:

STEP . 4 : Tokenize sentences.

STEP . 5 : Convert to lowercase.

STEP . 6 : Remove punctuation and stopwords.

STEP . 7 : Train Word2Vec Model:

STEP . 8 : Use Gensim's Word2Vec() with parameters like vector_size, window, and sg.

STEP . 9 : Save and Test Model:

STEP . 10 : Save the model using .save().

STEP . 11 : Retrieve word embeddings.

STEP . 12 : Find and display most similar words.

## PROGRAM:
```python
# Import necessary libraries
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample corpus
corpus = [
    "Natural language processing is a field of artificial intelligence.",
    "It enables computers to understand human language.",
    "Word embedding is a representation of words in a dense vector space.",
    "Gensim is a library for training word embeddings in Python.",
    "Machine learning and deep learning techniques are widely used in NLP."
]

# Preprocess the text: Tokenize, remove punctuation and stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

# Apply preprocessing to the corpus
processed_corpus = [preprocess_text(sentence) for sentence in corpus]

# Train a Word2Vec model
model = Word2Vec(sentences=processed_corpus, vector_size=100, window=2, min_count=1, sg=1)  # sg=1 uses Skip-gram

# Save the model for future use
model.save("word2vec_model.model")

# Test the model by finding the embedding of a word
word = "vector"
if word in model.wv:
    print(f"Embedding for '{word}':\n{model.wv[word]}")
else:
    print(f"'{word}' not found in vocabulary.")

# Find similar words
similar_words = model.wv.most_similar(word, topn=2)
print(f"Words similar to '{word}':")
for similar_word, similarity in similar_words:
    print(f"{similar_word}: {similarity:.4f}")
```
## OUTPUT :
![Screenshot 2025-06-04 203515](https://github.com/user-attachments/assets/0c184354-22fe-498b-9751-897d3e78d4fc)

## RESULT:
The Word2Vec model was successfully trained on a sample NLP corpus. The model was able to generate vector embeddings for words and retrieve semantically similar words.
