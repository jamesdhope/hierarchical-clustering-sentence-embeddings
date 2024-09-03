from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a larger set of sentences on various topics
sentences = [
    # Technology
    "Artificial intelligence is revolutionizing many industries.",
    "The latest smartphone models feature advanced camera systems.",
    "Quantum computing promises to solve complex problems faster.",
    "Cybersecurity threats are becoming increasingly sophisticated.",
    
    # Environment
    "Climate change is causing more frequent extreme weather events.",
    "Renewable energy sources are crucial for a sustainable future.",
    "Deforestation is a major contributor to biodiversity loss.",
    "Ocean pollution is harming marine ecosystems worldwide.",
    
    # Sports
    "The Olympics bring together athletes from around the world.",
    "Football remains the most popular sport globally.",
    "Tennis Grand Slams attract millions of viewers each year.",
    "E-sports are gaining recognition as legitimate competitive events.",
    
    # Food
    "Plant-based diets are becoming more popular for health reasons.",
    "Traditional cuisines often reflect local cultural heritage.",
    "Fast food chains are adapting to healthier eating trends.",
    "Molecular gastronomy combines science and culinary arts.",
    
    # Culture
    "Social media has transformed how people communicate and share information.",
    "Museums play a vital role in preserving historical artifacts.",
    "Music streaming services have changed the way we consume music.",
    "Global travel has increased cultural exchange and understanding.",
    
    # Overlapping Topics
    "Technology is being used to combat environmental issues.",
    "Sports nutrition has become a significant field of food science.",
    "Cultural diversity influences culinary traditions around the world.",
    "Virtual reality is changing how we experience sports and culture.",
    "Sustainable agriculture combines technology and environmental concerns.",
    "Social media influencers impact both culture and consumer behavior.",
    "Wearable technology is transforming both healthcare and fitness tracking.",
    "The gig economy is reshaping work culture in the digital age.",
    "Artificial intelligence is being applied in sports analytics and training.",
    "Smart cities integrate technology to improve urban environments."
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Perform hierarchical clustering using the 'ward' method
linked = linkage(embeddings, method='ward')

# Create a larger figure
plt.figure(figsize=(15, 15))

# Plot the dendrogram with improved readability
dendrogram(linked,
           orientation='left',  # Change orientation to left
           labels=sentences,
           distance_sort='descending',
           show_leaf_counts=True,
           leaf_font_size=10)  # Increase font size

# Adjust layout and labels
plt.title('Dendrogram of Sentence Embeddings', fontsize=20)
plt.xlabel('Distance', fontsize=16)
plt.ylabel('Sentences', fontsize=16)

# Rotate y-axis labels for better readability
plt.yticks(rotation=0, ha='right', va='center')

# Adjust subplot parameters to give specified padding
plt.subplots_adjust(left=0.4)  # Increase left margin for labels

# Improve overall layout
plt.tight_layout()

# Show the plot
plt.show()