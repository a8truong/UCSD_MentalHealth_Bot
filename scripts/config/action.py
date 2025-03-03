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
    Loads PDFs and splits them into smaller chunks using parallel processing.
    """
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
    If the message is a question, use the context to answer it. If not, use the context to make any suggestions.
    Remember to be empathetic and kind, considering you are talking to a UCSD student.
    Direct students to events related to their problem if possible and provide the description of the resources provided. 
    Please also include information on how to sign up for any events or access any resources mentioned.
    Please also ask if the student needs more information about the resources provided.

    Keep it simple.

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