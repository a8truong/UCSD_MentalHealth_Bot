import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from operator import itemgetter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

# Initialize OpenAI model and embeddings
def initialize_rag():
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
    parser = StrOutputParser()

    # Load and process documents
    loader = PyPDFLoader("data/flyer-rise.pdf")
    pages = loader.load_and_split()

    # Create vector store
    vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompt template
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    # Define the RAG chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    return chain
