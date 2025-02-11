from typing import Optional
from nemoguardrails.actions import action

import concurrent.futures
import os
import glob
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from operator import itemgetter

from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(pdf_path, text_splitter):
    """Process a single PDF file by loading and splitting it."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)
    return chunks

def load_multiple_pdfs(pdf_directory="data", chunk_size=500, chunk_overlap=100):
    """Loads PDFs and splits them into smaller chunks using parallel processing."""
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    all_pages = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    # Use ThreadPoolExecutor to parallelize PDF processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        results = list(executor.map(lambda pdf: process_pdf(pdf, text_splitter), pdf_files))
    
    # Flatten the list of chunks (since executor.map returns a list of lists)
    for result in results:
        all_pages.extend(result)

    return all_pages

def fetch_and_parse_url(url):
    """Fetches and parses the content of a web page."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            content = "\n".join(paragraphs)
            return Document(page_content=content, metadata={"source": url})
        else:
            print(f"Failed to fetch {url}")
            return None
    except requests.RequestException as e:
        print(f"Error while fetching {url}: {e}")
        return None

def scrape_multiple_websites(urls):
    """Scrapes multiple websites in parallel and returns the documents."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use executor.map to parallelize the scraping of multiple URLs
        results = list(executor.map(fetch_and_parse_url, urls))
    
    # Filter out any None results (failed fetches)
    return [result for result in results if result is not None]

import faiss
import numpy as np

@action(is_system_action=True)
async def retrieve(query: str) -> list:
    embeddings = OpenAIEmbeddings()

    # Load and process documents
    pages = load_multiple_pdfs("data")

    # List of URLs to scrape
    urls = [
        "https://caps.ucsd.edu/services/groups.html#Psychotherapy-and-Support-Group",
        "https://caps.ucsd.edu/services/letstalk.html",
        "https://caps.ucsd.edu/resources/iflourish.html#Headspace"
    ]

    # Scrape web pages
    web_documents = scrape_multiple_websites(urls)

    # Combine PDFs and Web Data
    all_documents = pages + web_documents

    # Convert documents to embeddings using embed_documents method
    document_texts = [doc.page_content for doc in all_documents]
    embeddings_list = embeddings.embed_documents(document_texts)

    # Convert embeddings to a numpy array (for FAISS)
    embeddings_array = np.array(embeddings_list).astype('float32')

    # Create FAISS index
    d = embeddings_array.shape[1]  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance index (you can choose another type of FAISS index)

    # Add embeddings to the FAISS index
    index.add(embeddings_array)

    k = 5
    query_embedding = embeddings.embed_query(query)  # Get query embedding
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, k)  # Perform search
    return [all_documents[i] for i in indices[0]]

async def rag(query: str, contexts: list) -> str:
    print("> RAG Called")  # we'll add this so we can see when this is being used
    context_str = "\n".join(contexts)
    # place query and contexts into RAG prompt
    template = """
    If the message is a question, use the context to answer it. If not, use the context to make any suggestions- remember you are supposed to be empathetic and kind, consider that you are talking
    to a UCSD student. Direct students to Community Connections events related to their problem if possible.
    
    Context: {context}

    Message: {question}
    """
    prompt = PromptTemplate.from_template(template)
    # generate answer
    res = openai.Completion.create(
        engine=MODEL,
        prompt=prompt,
        temperature=0.0,
        max_tokens=100
    )
    return res['choices'][0]['text']