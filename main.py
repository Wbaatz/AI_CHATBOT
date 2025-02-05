import warnings
warnings.filterwarnings('ignore')

# Install required libraries (run these commands in your terminal beforehand)
# pip install datasets pinecone-client fastapi uvicorn sentence-transformers tqdm torch
import google.generativeai as genai
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch
from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_KEY"))
# Load the FAQ data
with open('FAQS.json', 'r') as f:
 data = json.load(f)


# Extract questions and answers

def gemini_generate(prompt):
  model = genai.GenerativeModel("gemini-1.5-flash")
  response = model.generate_content(prompt)
  return response.text

questions = [item['question'] for item in data]
answers = [item['answer'] if isinstance(item['answer'], str) else " ".join(item['answer']) for item in data]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
 print('Sorry no cuda.')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_KEY")  # Replace with your Pinecone API key
INDEX_NAME =os.getenv("INDEX_NAME")

print(f"Pinecone API Key: {PINECONE_API_KEY}")
print(f"my index : {INDEX_NAME}")


pinecone = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
pinecone.create_index(
    name=INDEX_NAME,
    dimension=model.get_sentence_embedding_dimension(),
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
index = pinecone.Index(INDEX_NAME)



faq_embeddings = []
for faq in data:
    text = faq['question'] + " " + " ".join(faq['answer'])  # Convert list to string
    embedding = model.encode(text).tolist()  # Generate embedding

    # Ensure correct format for Pinecone
    faq_embeddings.append({
        "id": faq['id'],          # Unique ID
        "values": embedding,      # Embedding values
        "metadata": {             # Metadata for reference
            "question": faq['question'],
            "answer": " ".join(faq['answer'])
        }
    })
index.upsert(faq_embeddings)




def retrieve_docs(query, top_k=3):
    """Retrieve top K documents from Pinecone based on semantic similarity."""
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    return [match["metadata"]["answer"] for match in results["matches"]]


def refine_query(original_query, retrieved_docs):
    """Ask LLM to improve the query if the initial search is too vague."""
    prompt = f"""
    The user asked: "{original_query}"
    Retrieved documents: {retrieved_docs}

    If the retrieved results are not specific enough, suggest a refined query.
    Otherwise, return the original query.
    """
    refined_query = gemini_generate(prompt)
    return refined_query["choices"][0]["text"].strip()

def multi_step_retrieval(query):
    """Perform retrieval in multiple steps to get the best context for RAG."""
  
    context = retrieve_docs(query)
    if not context or len(context) < 2: 
        query = refine_query(query, context)
        context = retrieve_docs(query)

    return context



def rag_qa(query):
    """Retrieve relevant documents and generate an answer using LLM."""
    context = multi_step_retrieval(query)

    full_prompt = f"""
Use the following information to answer the question concisely:

{context}

Q: {query}  
A: Provide the answer  without introductory phrases like "According to the provided text." but do give a description 
"""
    response = gemini_generate(full_prompt)

    print(response)

    return response.strip()


# FastAPI app
app = FastAPI()

# Pydantic model for query input
class QueryInput(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(input: QueryInput):
    try:
        query_text = input.query
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        # Run the query
        response = rag_qa(query_text)
        return {"results": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

