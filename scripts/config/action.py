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

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from openai import OpenAI

import faiss
import numpy as np
import time
import json
from sentence_transformers import SentenceTransformer
import torch
torch.classes.__path__ = []
torch.set_num_threads(1)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(pdf_path, text_splitter):
    """
    Process a single PDF file by loading and splitting it.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)
    return chunks

def load_multiple_pdfs(pdf_directory="data", chunk_size=500, chunk_overlap=100):
    """
    Loads PDFs and splits them into smaller chunks. Uses cached data if available.
    """
    # Try to load cached PDF data
    pdf_documents = load_pdf_data()
    if pdf_documents:
        print("Loaded PDF data from cache.")
        return pdf_documents  # Use cached data

    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    all_pages = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda pdf: process_pdf(pdf, text_splitter), pdf_files))
    
    for result in results:
        all_pages.extend(result)

    # Save processed PDFs to cache
    save_pdf_data(all_pages)

    return all_pages

PDF_CACHE_FILE = "scripts/config/kb/pdf_data.json"

def save_pdf_data(documents):
    """Save processed PDF data to a JSON file."""
    data = [{"source": doc.metadata["source"], "content": doc.page_content} for doc in documents]
    with open(PDF_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_pdf_data():
    """Load processed PDF data from JSON file."""
    if os.path.exists(PDF_CACHE_FILE):
        with open(PDF_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=item["content"], metadata={"source": item["source"]}) for item in data]
    return []

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

CACHE_FILE = "scripts/config/kb/scraped_data.json"

def save_scraped_data(documents):
    """Save scraped documents to a JSON file."""
    data = [{"source": doc.metadata["source"], "content": doc.page_content} for doc in documents]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_scraped_data():
    """Load previously scraped data from JSON file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=item["content"], metadata={"source": item["source"]}) for item in data]
    return []

def init_cache():
    index = faiss.IndexFlatL2(768)
    if index.is_trained:
        print("Index trained")

    # Initialize Sentence Transformer model
    encoder = SentenceTransformer("all-mpnet-base-v2")

    return index, encoder

def retrieve_cache(json_file):
    if not os.path.exists(json_file):  # If file doesn't exist, create it
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}
        with open(json_file, "w") as file:
            json.dump(cache, file, indent=4)  # Write the default structure to file
        return cache

    try:
        with open(json_file, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Warning: {json_file} is corrupted. Resetting cache.")
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}
        with open(json_file, "w") as file:
            json.dump(cache, file, indent=4)  # Overwrite the corrupted file
        return cache

def store_cache(json_file, cache):
    # Convert Document objects to a dictionary format
    def convert_to_serializable(obj):
        if isinstance(obj, Document):
            return obj.__dict__  # Convert to dictionary
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(json_file, "w") as file:
        json.dump(cache, file, default=convert_to_serializable, indent=4)

class semantic_cache:
    def __init__(self, json_file="cache_file.json", thresold=0.35, max_response=100, eviction_policy=None):
        """Initializes the semantic cache.

        Args:
        json_file (str): The name of the JSON file where the cache is stored.
        thresold (float): The threshold for the Euclidean distance to determine if a question is similar.
        max_response (int): The maximum number of responses the cache can store.
        eviction_policy (str): The policy for evicting items from the cache.
                                This can be any policy, but 'FIFO' (First In First Out) has been implemented for now.
                                If None, no eviction policy will be applied.
        """

        # Initialize Faiss index with Euclidean distance
        self.index, self.encoder = init_cache()

        # Set Euclidean distance threshold
        # a distance of 0 means identicals sentences
        # We only return from cache sentences under this thresold
        self.euclidean_threshold = thresold

        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.eviction_policy = eviction_policy

    def evict(self):
        """Evicts an item from the cache based on the eviction policy."""
        if self.eviction_policy and len(self.cache["questions"]) > self.max_response:
            for _ in range((len(self.cache["questions"]) - self.max_response)):
                if self.eviction_policy == "FIFO":
                    self.cache["questions"].pop(0)
                    self.cache["embeddings"].pop(0)
                    self.cache["answers"].pop(0)
                    self.cache["response_text"].pop(0)

    async def ask(self, question: str) -> str:
        # Method to retrieve an answer from the cache or generate a new one
        start_time = time.time()
        try:
            # First we obtain the embeddings corresponding to the user question
            embedding = self.encoder.encode([question])

            # Search for the nearest neighbor in the index
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)
            # print(I[0][0])
            # print(D[0][0])

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])

                    # print("Answer recovered from Cache. ")
                    # print(f"{D[0][0]:.3f} smaller than {self.euclidean_threshold}")
                    # print(f"Found cache in row: {row_id} with score {D[0][0]:.3f}")
                    # print(f"response_text: " + self.cache["response_text"][row_id])

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    #print(self.cache["answers"][row_id])
                    return self.cache["answers"][row_id]

            # Handle the case when there are not enough results
            # or Euclidean distance is not met, call retrieve
            retrieved_docs = self.retrieve(question)
            response_text = retrieved_docs[0].page_content

            self.cache["questions"].append(question)
            self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append(retrieved_docs)
            self.cache["response_text"].append(response_text)

            # print("Answer recovered from retrieve. ")
            # print(f"response_text: {response_text}")

            self.index.add(embedding)

            self.evict()

            store_cache(self.json_file, self.cache)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return retrieved_docs
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")
        
    def retrieve(self, query: str) -> list:
        print('Hi, retrieve yo mamma')
        embeddings = OpenAIEmbeddings()

        # Load and process documents
        pages = load_multiple_pdfs("data")

        # Try to load cached web data, otherwise scrape and save
        web_documents = load_scraped_data()
        if not web_documents:  # If cache is empty, scrape
            urls = [
                "https://caps.ucsd.edu/services/groups.html#Psychotherapy-and-Support-Group",
                "https://caps.ucsd.edu/services/letstalk.html",
                "https://caps.ucsd.edu/resources/iflourish.html#Headspace"
            ]
            web_documents = scrape_multiple_websites(urls)
            save_scraped_data(web_documents)  # Save new data to cache

        # Combine PDFs and Web Data
        all_documents = pages + web_documents

        # Apply Semantic Chunking**
        chunk_size = 500  # Customize based on document size
        chunk_overlap = 100  # Ensure overlap to preserve context
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunked_docs = []
        for doc in all_documents:
            chunks = splitter.split_text(doc.page_content)  # Break into semantic chunks
            chunked_docs.extend([Document(page_content=chunk) for chunk in chunks])  # Wrap in Document objects

        # Convert documents to embeddings using embed_documents method
        embeddings_list = embeddings.embed_documents([doc.page_content for doc in chunked_docs])

        # Convert embeddings to a numpy array (for FAISS)
        embeddings_array = np.array(embeddings_list).astype('float32')

        # Create FAISS index
        d = embeddings_array.shape[1]  # Embedding dimension
        if os.path.exists("faiss_index.bin"):
            index = faiss.read_index("faiss_index.bin")
        else:
            index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings_array)  # Normalize for Cosine Similarity
        index.add(embeddings_array)  # Add embeddings

        # Save index for reuse
        faiss.write_index(index, "faiss_index.bin")

        k = min(5, embeddings_array.shape[0])  # Ensure k is within range
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_embedding, k)  # Perform search

        retrieved_docs = [chunked_docs[i] for i in indices[0] if i < len(chunked_docs)]
        return retrieved_docs

@action(is_system_action=True)
async def rag(query: str, contexts: list) -> str:
    print("> RAG Called")  
    relevant_chunks = "\n".join([
    doc.page_content if isinstance(doc, Document) else doc["page_content"]
    for doc in contexts
])
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)

    # place query and contexts into RAG prompt
    TEMPLATE = """
    You are a UCSD mental health assistant. Your goal is to recommend resources based on the provided context.

    If the context does not contain a relevant resource, say: "I couldn't find any relevant information, but here are general support options..."

    Guidelines:
    1. Be **empathetic** and **kind**.
    2. Recommend **only new** resources that haven't been mentioned before.
    3. Provide **event sign-up details**, if available.
    4. Break responses into **clear paragraphs**.

    Context:
    {context}

    Query: {question}

    Answer:
    """
    prompt_template = PromptTemplate.from_template(TEMPLATE)
    input_variables = {"question": query, "context": relevant_chunks}

    # Generate response using LangChain pipeline
    output_parser = StrOutputParser()
    chain = prompt_template | model | output_parser
    answer = await chain.ainvoke(input_variables)

    return answer

async def concern(query: str) -> bool:
    """
    Uses an LLM to check if the prompt contains more than one sentence
    and if it discusses mental health concerns such as stress, anxiety, or emotional distress.

    Args:
        prompt (str): The input prompt to analyze.

    Returns:
        bool: True if the prompt is a mental health concern with multiple sentences.
    """
    #print('is mental health concern?')
    # Call the LLM to analyze the prompt
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the desired LLM model
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that detects if a message describes a mental health concern."
            },
            {
                "role": "user",
                "content": f"Does the following message contain mental health concerns and multiple sentences? {query}"
            }
        ],
        max_tokens=150,
        temperature=0.5
    )

    result = response.choices[0].message.content.strip().lower()
    # Check the response from the model to determine if it recognizes multiple sentences and mental health concerns
    if "yes" in result and "multiple sentences" in result and "mental health" in result:
        return True
    return False