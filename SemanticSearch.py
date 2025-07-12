import pandas as pd

# Sample clothing items
data = {
    "item_id": [1, 2, 3, 4],
    "description": [
        "Warm down jacket perfect for winter",
        "Light cotton hoodie for spring",
        "Elegant evening gown",
        "Thick parka suitable for snow"
    ]
}

df = pd.DataFrame(data)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Light, fast model

# Generate embeddings for item descriptions
item_embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)


import faiss
import numpy as np

# Convert embeddings to float32
item_embeddings = np.array(item_embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(item_embeddings.shape[1])
index.add(item_embeddings)


#SEARCH FUNCTION
def smart_search(query, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    
    print(f"\nResults for: '{query}'\n")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {df.iloc[idx]['description']} (Item ID: {df.iloc[idx]['item_id']})")
        
#TestCase
smart_search("winter coat")
smart_search("party dress")
smart_search("spring outfit")