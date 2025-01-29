import warnings
warnings.filterwarnings('ignore')

# Install required libraries (run these commands in your terminal beforehand)
# pip install datasets pinecone-client fastapi uvicorn sentence-transformers tqdm torch

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

# Load the FAQ data
with open('FAQS.json', 'r') as f:
    data = json.load(f)

# Extract questions and answers
questions = [item['question'] for item in data]
answers = [item['answer'] if isinstance(item['answer'], str) else " ".join(item['answer']) for item in data]

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('CUDA is not available, using CPU.')

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_KEY")  # Replace with your Pinecone API key
INDEX_NAME =os.getenv("INDEX_NAME")

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

# Batch process and upsert data to Pinecone
batch_size = 7
for i in tqdm(range(0, len(questions), batch_size)):
    # Determine batch range
    i_end = min(i + batch_size, len(questions))
    # Create batch data
    ids = [str(j) for j in range(i, i_end)]
    metadatas = [{'question': questions[j], 'answer': answers[j]} for j in range(i, i_end)]
    embeddings = model.encode(questions[i:i_end])
    records = zip(ids, embeddings, metadatas)
    # Upsert batch to Pinecone
    index.upsert(vectors=records)

print("Data has been successfully upserted to Pinecone!")

# Describe index stats
stats = index.describe_index_stats()
print(f"Index Stats: {stats}")

# Helper function to query Pinecone
def run_query(query: str):
    embedding = model.encode(query).tolist()
    results = index.query(
        top_k=10,
        vector=embedding,
        include_metadata=True,
        include_values=False
    )
    # return [
    #     {
    #         'score': round(result['score'], 2),
    #         'question': result['metadata'].get('question', 'Unknown question'),
    #         'answer': result['metadata'].get('answer', 'No answer provided')
    #     }
    #     for result in results['matches']
    # ]


    return max(
    (result['metadata'].get('answer', 'No answer provided') for result in results['matches']),
    key=lambda answer: next(
        result['score']
        for result in results['matches']
        if result['metadata'].get('answer') == answer
    )
    )


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
        response = run_query(query_text)
        return {"results": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

