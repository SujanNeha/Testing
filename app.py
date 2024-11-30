import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json

# Load structured data
with open("structured_data.json", "r") as f:
    data = json.load(f)

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each section
corpus = [section["content"] for section in data["sections"]]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Load a pre-trained question-answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to find the most relevant section
def find_relevant_section(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
    best_match_idx = scores.argmax().item()
    return corpus[best_match_idx]

# Extract answer from the relevant section
def extract_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Streamlit app
st.title("BSRIA Rules of Thumb Q&A")
st.write("Ask any question about the BSRIA Rules of Thumb document.")

query = st.text_input("Enter your question:")
if query:
    relevant_section = find_relevant_section(query)
    answer = extract_answer(query, relevant_section)
    st.write("**Answer:**", answer)
    st.write("**Relevant Section:**", relevant_section)
