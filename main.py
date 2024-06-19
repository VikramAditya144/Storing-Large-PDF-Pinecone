import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv()

# Function to extract text from a PDF

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Function to chunk the extracted text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks



pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Pinecone API key is not set in the environment variables")


# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Create a Pinecone index
index_name = "book-concepts"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Assuming the dimension of the embeddings
        metric='euclidean',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
# Load pre-trained model and tokenizer for embeddings
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings from text using DistilBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Pooling method: mean
    return embeddings.tolist()

# Function to store embeddings in Pinecone
def store_embeddings(chunks):
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            index.upsert([(str(i), embedding)])

# Function to process the PDF
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    store_embeddings(chunks)

pdf_path = "/Users/vikramaditya./Desktop/Enhansa/mvp/AtomicHabits.pdf"
process_pdf(pdf_path)


