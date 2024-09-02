from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def get_cluster_keywords(sentences, n=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    vectorizer = CountVectorizer().fit(sentences)
    vocab = vectorizer.get_feature_names_out()
    word_embeddings = model.encode(vocab)
    cluster_embedding = np.mean(embeddings, axis=0)
    similarities = cosine_similarity([cluster_embedding], word_embeddings)[0]
    top_n_idx = similarities.argsort()[-n:][::-1]
    top_keywords = [vocab[i] for i in top_n_idx]
    return top_keywords

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define sentences
sentences = [
    "Artificial intelligence is revolutionizing industries.",
    "Smartphone models feature advanced camera systems.",
    "Quantum computing solves complex problems faster.",
    "Cybersecurity threats are increasingly sophisticated.",
    "Climate change causes extreme weather events.",
    "Renewable energy is crucial for sustainability.",
    "Deforestation contributes to biodiversity loss.",
    "Ocean pollution harms marine ecosystems.",
    "The Olympics bring together global athletes.",
    "Football remains the most popular sport.",
    "Tennis Grand Slams attract millions of viewers.",
    "E-sports gain recognition as competitive events.",
    "Plant-based diets become more popular.",
    "Traditional cuisines reflect cultural heritage.",
    "Fast food chains adapt to healthier trends.",
    "Molecular gastronomy combines science and culinary arts."
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Perform hierarchical clustering
linked = linkage(embeddings, method='ward')

# Create a larger figure
plt.figure(figsize=(20, 10))

# Plot the dendrogram
dendrogram(
    linked,
    orientation='left',
    labels=sentences,
    leaf_rotation=0,
    leaf_font_size=8,
)

# Adjust layout and labels
plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
plt.xlabel('Distance', fontsize=16)
plt.ylabel('Sentences', fontsize=16)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Cut the dendrogram into a specified number of clusters
num_clusters = 5
cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

# Group sentences by cluster
clusters = defaultdict(list)
for sentence, label in zip(sentences, cluster_labels):
    clusters[label].append(sentence)

# Print keywords for each cluster
print("\nCluster Keywords:")
for cluster_id, cluster_sentences in clusters.items():
    keywords = get_cluster_keywords(cluster_sentences, n=3)
    print(f"\nCluster {cluster_id}:")
    print(f"  Keywords: {', '.join(keywords)}")
    print(f"  Sentences:")
    for sentence in cluster_sentences:
        print(f"    - {sentence}")