import os
import glob
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from operator import itemgetter

from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

config = RailsConfig.from_path("./config")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

#Load and split pages from multiple PDF files in a given directory.
def load_multiple_pdfs(pdf_directory="data"):
    
    # Use glob to find all PDF files in the specified directory
    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    
    all_pages = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        all_pages.extend(pages)
    
    return all_pages

# Initialize OpenAI model and embeddings
def initialize_rag():
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
    parser = StrOutputParser()

    # Load and process documents
    pages = load_multiple_pdfs("data")

    # Create vector store
    vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompt template
    template = """
    Answer the question based on the context below.

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    guardrails = RunnableRails(config)

    # Define the RAG chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | (guardrails | model)
        | parser
    )

    return chain