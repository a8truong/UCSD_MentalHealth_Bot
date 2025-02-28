import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'scripts')))
from config.action import semantic_cache, rag

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

cache = semantic_cache()
retrieve = cache.retrieve

# Load Guardrails configuration
config = RailsConfig.from_path("../scripts/config")
rails = LLMRails(config)
rails.register_action(action=retrieve, name="retrieve")
rails.register_action(action=rag, name="rag")

# Number of trials to run per prompt
iterations = 5

async def process_prompt(prompt, iterations=5):
    """Processes a single prompt multiple times using both Guardrails and direct RAG."""

    # Generate async objects to run multiple requests concurrently for ea. prompt
    guardrails_tasks = [rails.generate_async(prompt) for _ in range(iterations)]
    rag_tasks = [asyncio.to_thread(retrieve, prompt) for _ in range(iterations)]
    
    # Wait for all tasks to complete concurrently
    guardrails_responses = await asyncio.gather(*guardrails_tasks)
    contexts = await asyncio.gather(*rag_tasks)
    rag_responses = await asyncio.gather(*(rag(prompt, context) for context in contexts))
    
    return guardrails_responses, rag_responses


async def process_csv(input_csv, output_csv):
    """Reads a CSV file with prompts and outputs both Guardrails and RAG responses."""

    df = pd.read_csv(input_csv)  # Load CSV
    
    # Initialize lists to store results
    all_prompts = []
    all_guardrails_responses = []
    all_rag_responses = []
    
    # Process all prompts asynchronously and collect results
    for prompt in df["Prompt"]:
        guardrails_responses, rag_responses = await process_prompt(prompt)
        
        # Repeat the prompt 5 times and append corresponding responses
        for i in range(iterations):
            all_prompts.append(prompt)
            all_guardrails_responses.append(guardrails_responses[i])
            all_rag_responses.append(rag_responses[i])
    
    # Create a DataFrame where each prompt has 5 corresponding responses
    result_df = pd.DataFrame({
        "Prompt": all_prompts,
        "Guardrails_Response": all_guardrails_responses,
        "Direct_RAG_Response": all_rag_responses
    })
    
    # Save the results to a new CSV
    result_df.to_csv(output_csv, index=False)

# Run the async function
asyncio.run(process_csv("./input_prompts.csv", "./output_responses.csv"))
