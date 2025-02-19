from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM 
from langchain.prompts import ChatPromptTemplate
import argparse

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
    Answer the following question based on only the given conext:
    
    {context}

    -----

    Answer the question based on the above context: {question}
"""

ollama_embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    num_gpu=-1
    )

def main():
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text


    # connecting to the chroma db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=ollama_embedding)
    
    # search through the db
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        print("Unable to find the relevant data.")
        return
    
    context_text = "\n---\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    print(prompt)

    model = OllamaLLM(
        model="deepseek-r1:1.5b", 
        base_url="http://localhost:11434", 
        num_gpu=-1
        )

    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()