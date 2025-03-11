import os
import faiss
import json
import numpy as np
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults

# Load environment variables
load_dotenv(find_dotenv())
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")

# Initialize LLM once to avoid re-instantiating in functions
model_name = 'llama-3.3-70b-versatile'
llm = ChatGroq(api_key=GROQ_API, model_name=model_name, temperature=0.0)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# FAISS Index
dimension = 1536  # OpenAI's embedding dimension
index = faiss.IndexFlatL2(dimension)
documents = []  # Store document texts separately


def search_tavily(query: str, max_results: int = 5) -> dict:
    """
    Searches for relevant articles using TavilySearchResults.
    """
    tool = TavilySearchResults(max_results=max_results)
    response_json = tool.invoke({"query": query})
    print(f"Response JSON from SERP: {response_json}")
    return response_json


def pick_best_articles_urls(response_json: dict, query: str) -> list:
    """
    Uses an LLM to select the best articles from search results and return a list of URLs.
    """
    response_str = json.dumps(response_json)

    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"],
        template="""
          You are a world-class journalist, researcher, and tech expert.
          You excel at selecting the most relevant and high-quality articles.

          SEARCH RESULTS: {response_str}

          QUERY: {query}

          Select the best 3 articles and return ONLY an array of their URLs.
          If a URL is invalid, replace it with 'www.google.com'.

          Return ONLY a JSON array with the URLs.
        """
    )

    article_chooser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    try:
        urls_str = article_chooser_chain.run(response_str=response_str, query=query)
        url_list = json.loads(urls_str)
    except json.JSONDecodeError:
        print("Warning: LLM did not return valid JSON. Returning an empty list.")
        url_list = []

    return ["https://www.google.com" if not url.startswith("http") else url for url in url_list]


def extract_content_from_urls(urls: list):
    """
    Loads and processes content from URLs, then stores it in a FAISS vector database.
    """
    global index, documents

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    docs = text_splitter.split_documents(data)

    # Convert text into embeddings and store in FAISS
    text_embeddings = []
    for doc in docs:
        embedding = embeddings.embed_query(doc.page_content)
        text_embeddings.append(embedding)
        documents.append(doc.page_content)  # Store actual text

    text_embeddings = np.array(text_embeddings, dtype='float32')
    
    if len(text_embeddings) > 0:
        index.add(text_embeddings)  # Add to FAISS index

    return index  # Return FAISS index


def summarizer(query: str, k: int = 4) -> str:
    """
    Retrieves relevant document chunks from FAISS and summarizes them using LLM.
    """
    global index, documents

    query_embedding = np.array([embeddings.embed_query(query)], dtype='float32')

    # Perform similarity search in FAISS
    _, nearest_indices = index.search(query_embedding, k)
    
    retrieved_docs = [documents[i] for i in nearest_indices[0] if i < len(documents)]
    docs_page_content = " ".join(retrieved_docs) if retrieved_docs else "No relevant content found."

    prompt_template = PromptTemplate(
        input_variables=["docs", "query"],
        template="""
           {docs}
           You are a top journalist and researcher. Write a concise and engaging newsletter summary about {query}.
           Ensure that:
             1) The content is informative and engaging.
             2) The length is appropriate for a newsletter.
             3) Insights, practical advice, and links (if necessary) are included.
        """
    )

    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    response = summarizer_chain.run(docs=docs_page_content, query=query)

    return response.replace("\n", " ")


def generate_newsletter(summaries: str, query: str) -> str:
    """
    Generates a final newsletter text based on the summarized content.
    """
    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"],
        template="""
        {summaries_str}
        Create a newsletter about {query} in a concise and engaging style similar to "5-Bullet Friday".
        - Start with "Hi All!"
        - Provide an engaging introduction before the main content.
        - Keep it short and informative.
        - Offer practical tips, advice, and relevant links.
        - End with a wise quote and sign off as:
          "Eng. Ibrahim Alobaid"
        """
    )

    news_letter_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    return news_letter_chain.predict(summaries_str=summaries, query=query)


def generate_tech_newsletter(query: str) -> str:
    """
    Automates the workflow to generate a technology newsletter:
      1. Searches for relevant articles.
      2. Picks the best articles using an LLM.
      3. Extracts and processes content into FAISS.
      4. Summarizes the content.
      5. Generates a final newsletter.
    """
    serp_results = search_tavily(query)
    urls = pick_best_articles_urls(serp_results, query)
    
    if not urls:
        return "No suitable articles found. Try a different query."

    extract_content_from_urls(urls)
    summary_text = summarizer(query, k=4)
    return generate_newsletter(summary_text, query)


def main():
    st.set_page_config(page_title="Researcher...", 
                       page_icon=":parrot:", 
                       layout="wide")
    
    st.header("Generate a Newsletter :parrot:")
    query = st.text_input("Enter a topic...")

    if query:
        with st.spinner(f"Generating newsletter for {query}..."):
            search_results = search_tavily(query=query)
            urls = pick_best_articles_urls(response_json=search_results, query=query)
            extract_content_from_urls(urls)
            summaries = summarizer(query)
            newsletter_thread = generate_newsletter(summaries, query)

            with st.expander("Search Results"):
                st.json(search_results)
            with st.expander("Best URLs"):
                st.json(urls)
            with st.expander("Summaries"):
                st.info(summaries)
            with st.expander("Generated Newsletter"):
                st.info(newsletter_thread)

        st.success("Newsletter generation complete! 🎉")


if __name__ == '__main__':
    main()
