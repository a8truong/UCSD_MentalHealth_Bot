import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'scripts')))
from config.action import retrieve, rag

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load Guardrails configuration
config = RailsConfig.from_path("./scripts/config")
rails = LLMRails(config)
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=rag, name="rag")

def process_prompt(prompt):
    """Processes a single prompt using both Guardrails and direct RAG."""
    
    # Guardrails response
    guardrails_response = rails.generate(prompt)
    
    # Direct RAG response
    contexts = asyncio.run(retrieve(prompt))  # Retrieve relevant contexts
    direct_rag_response = asyncio.run(rag(prompt, contexts))
    
    return guardrails_response, direct_rag_response

def process_csv(input_csv, output_csv):
    """Reads a CSV file with prompts and outputs both Guardrails and RAG responses."""
    df = pd.read_csv(input_csv)  # Load CSV
    
    # Apply function to each prompt and store both responses
    df[["Guardrails_Response", "Direct_RAG_Response"]] = df["Prompt"].apply(
        lambda prompt: pd.Series(process_prompt(prompt))
    )
    
    df.to_csv(output_csv, index=False)  # Save to new CSV

process_csv("./test/input_prompts.csv", "./test/output_responses.csv")