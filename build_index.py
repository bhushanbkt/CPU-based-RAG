from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Sample document corpus
documents = [
    "Pulmonary embolism is a condition in which one or more arteries in the lungs become blocked by a blood clot.",
    "CT pulmonary angiography is the gold standard imaging test to diagnose PE.",
    "Common symptoms of PE include shortness of breath, chest pain, and rapid heart rate.",
    "Blood thinners like heparin or warfarin are often prescribed to treat PE.",
    "Prevention of PE includes staying active, wearing compression stockings, and avoiding prolonged sitting."
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)

# Save documents
np.save("docs.npy", np.array(documents))

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, "doc_index.faiss")
print("FAISS index and documents saved.")
