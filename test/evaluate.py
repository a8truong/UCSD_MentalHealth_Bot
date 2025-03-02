import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'scripts')))
from config.action import semantic_cache, rag, concern

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load Guardrails configuration
config = RailsConfig.from_path("../scripts/config")
rails = LLMRails(config)
# Cache for semantic search
cache = semantic_cache("4cache.json")
ask = cache.ask
rails.register_action(action=cache.ask, name="ask")
rails.register_action(action=rag, name="rag")
rails.register_action(action=concern, name="concern")

# Number of trials to run per prompt
iterations = 5

async def process_prompt_baseline(prompt, iterations=5):
    # Define RAG/cache retrieval tasks
    retrieval_tasks = [ask(prompt) for _ in range(iterations)]
    # RAG retrieval and response time
    rag_start = time.time()
    # Wait for all tasks to complete concurrently
    contexts = await asyncio.gather(*retrieval_tasks)
    # Define RAG generation tasks
    rag_generation_tasks = [rag(prompt, context) for context in contexts] 
    # Run RAG response generation
    rag_responses = await asyncio.gather(*rag_generation_tasks)
    rag_time = time.time() - rag_start
    print("Total baseline time (for 5 responses): ", rag_time)
    return rag_responses

async def process_prompt_healthbot(prompt, iterations=5):
    """Processes a single prompt multiple times using both Guardrails model"""

    # Generate async objects to run multiple requests concurrently for ea. prompt
    # Define guardrail tasks
    guardrails_tasks = [rails.generate_async(prompt) for _ in range(iterations)]
    # Get guardrail responses and time it
    guard_start = time.time()
    guardrails_responses = await asyncio.gather(*guardrails_tasks)
    guard_time = time.time() - guard_start
    print("Total MentalHealthBot time (for 5 responses): ", guard_time)
    return guardrails_responses

async def process_csv(input_csv, output_csv):
    """Reads a CSV file with prompts and outputs both Guardrails and RAG responses."""

    df = pd.read_csv(input_csv)  # Load CSV
    
    # Initialize lists to store results
    all_prompts = []
    all_guardrails_responses = []
    all_rag_responses = []
    
    # Process all prompts asynchronously and collect results
    for prompt in df["Prompt"]:
        #rag_responses = await process_prompt_baseline(prompt, iterations)
        #guardrails_responses = await process_prompt_healthbot(prompt, iterations)
        rag_responses = await process_prompt_baseline(prompt, iterations)
        
        # Repeat the prompt 5 times and append corresponding responses
        all_prompts.extend([prompt] * iterations)
        # all_guardrails_responses.extend(guardrails_responses)
        all_guardrails_responses.extend([""] * iterations)
        all_rag_responses.extend(rag_responses)
        #all_rag_responses.extend([""] * iterations)
    
    # Create a DataFrame where ea. observation has a prompt, Guardrails response, and RAG response
    result_df = pd.DataFrame({
        "Prompt": all_prompts,
        "Guardrails_Response": all_guardrails_responses,
        "Direct_RAG_Response": all_rag_responses
    })
    
    # Save the results to a new CSV
    result_df.to_csv(output_csv, index=False)

# Run the async function
asyncio.run(process_csv("./input_prompts.csv", "./output_responses.csv"))
