from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to get top keywords from a cluster of sentences
def get_cluster_keywords(sentences, n=3):
    vectorizer = CountVectorizer().fit(sentences)
    vocab = vectorizer.get_feature_names_out()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    word_embeddings = model.encode(vocab)
    sentence_embeddings = model.encode(sentences)
    cluster_embedding = np.mean(sentence_embeddings, axis=0)
    similarities = cosine_similarity([cluster_embedding], word_embeddings)[0]
    top_n_idx = similarities.argsort()[-n:][::-1]
    top_keywords = [vocab[i] for i in top_n_idx]
    return top_keywords

# Define the sentences
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

# Load the model and generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

# Perform hierarchical clustering
linked = linkage(embeddings, method='ward')

# Create a figure with adjusted size to accommodate the legend
fig = plt.figure(figsize=(15, 8))  # Increased width to allow space for the legend
ax_dendrogram = fig.add_axes([0.25, 0.1, 0.45, 0.8])  # Adjusted width to make space for legend
ax_dendrogram.set_title('Hierarchical Clustering Dendrogram', fontsize=10)
ax_dendrogram.set_xlabel('Distance', fontsize=8)
ax_dendrogram.set_ylabel('Sentences', fontsize=8)

# Plot the dendrogram
dendrogram(linked, orientation='left', labels=[f"{i}: {sent}" for i, sent in enumerate(sentences)], ax=ax_dendrogram)

# Track the forks and clusters
forks = {}
legend_text = ""
previous_clusters = []

for num_clusters in range(1, len(sentences) + 1):
    clusters = fcluster(linked, t=num_clusters, criterion='maxclust')
    
    # Identify unique clusters at this level
    cluster_indices = [sorted(np.where(clusters == i)[0].tolist()) for i in np.unique(clusters)]
    
    # Track new splits by comparing with previous clusterings
    new_forks = []
    for cluster in cluster_indices:
        if len(cluster) > 1 and cluster not in previous_clusters:  # Only include clusters with more than one sentence
            keywords = get_cluster_keywords([sentences[i] for i in cluster], n=1)
            keywords_str = ', '.join(keywords)
            fork_text = f"Fork {num_clusters - 1} (clusters={num_clusters}): {keywords_str} ({', '.join(map(str, cluster))})\n"
            new_forks.append(fork_text)
    
    if new_forks:
        legend_text += ''.join(new_forks)
        previous_clusters = cluster_indices  # Update the previous clusters to the current level

# Print the legend elements in the console
print("Legend Elements:")
print(legend_text)

# Add the legend text to the left side of the plot
fig.text(0.01, 0.5, legend_text, fontsize=7, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

# Adjust layout to ensure the plot and legend fit nicely
plt.subplots_adjust(left=0.55, right=0.95)

plt.show()
