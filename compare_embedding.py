import os
import asyncio
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain.evaluation import load_evaluator
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# If openAI API is available
# load_dotenv()

ollama_embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    num_gpu=-1
    )

# Using OpenAI API
# def calculate_evaluation():
#     embedding_function = OpenAIEmbeddings()
#     evaluator = load_evaluator("pairwise_embedding_distance")
#     words = ("apple", "iphone")
#     x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
#     print(f"Comparing ({words[0]}, {words[1]}): {x}")

async def main():
    embedding_func = ollama_embedding

    words = ("apple","iphone")

    vector1 = await embedding_func.aembed_query(words[0])
    vector2 = await embedding_func.aembed_query(words[1])

    vec1 = np.array(vector1)
    vec2 = np.array(vector2)

    # Compute cosine similarity
    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Compute cosine distance (1 - similarity)
    cosine_distance = 1 - cosine_similarity

    # print(f"Vector for '{words[0]}': {vector1}")
    print(f"Vector length: {len(vector1)}")
    # print(f"Vector for '{words[1]}': {vector2}")
    print(f"Vector length: {len(vector2)}")
    print(f"Cosine similarity between {words[0]} and {words[1]}: {cosine_similarity}")
    print(f"Cosine distance between {words[0]} and {words[1]}: {cosine_distance}")


asyncio.run(main())