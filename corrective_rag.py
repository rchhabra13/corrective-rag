"""Corrective RAG Agent with multi-stage workflow using LangGraph.

This module implements a sophisticated retrieval-augmented generation system
that combines document retrieval, relevance grading, query transformation, and
web search to provide comprehensive and accurate responses.
"""

import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import nest_asyncio
import streamlit as st
import yaml
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Global retriever variable
retriever: Optional[Any] = None


def initialize_session_state() -> None:
    """Initialize session state variables for API keys and URLs."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.anthropic_api_key = ""
        st.session_state.openai_api_key = ""
        st.session_state.tavily_api_key = ""
        st.session_state.qdrant_api_key = ""
        st.session_state.qdrant_url = "http://localhost:6333"
        st.session_state.doc_url = "https://arxiv.org/pdf/2307.09288.pdf"


def setup_sidebar() -> None:
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        st.session_state.anthropic_api_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.anthropic_api_key,
            type="password",
            help="Required for Claude 3 model",
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
        )
        st.session_state.tavily_api_key = st.text_input(
            "Tavily API Key",
            value=st.session_state.tavily_api_key,
            type="password",
        )
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password",
        )
        st.session_state.doc_url = st.text_input(
            "Document URL",
            value=st.session_state.doc_url,
        )

        if not all(
            [
                st.session_state.openai_api_key,
                st.session_state.anthropic_api_key,
                st.session_state.qdrant_url,
            ]
        ):
            st.warning("Please provide the required API keys and URLs")
            st.stop()

        st.session_state.initialized = True


initialize_session_state()
setup_sidebar()

# Use session state variables
openai_api_key = st.session_state.openai_api_key
tavily_api_key = st.session_state.tavily_api_key
anthropic_api_key = st.session_state.anthropic_api_key

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=st.session_state.openai_api_key,
)

# Initialize Qdrant client
client = QdrantClient(
    url=st.session_state.qdrant_url,
    api_key=st.session_state.qdrant_api_key,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_tavily_search(tool: TavilySearchResults, query: str) -> List[Dict[str, Any]]:
    """Execute web search with retry logic.

    Args:
        tool (TavilySearchResults): The Tavily search tool instance.
        query (str): The search query.

    Returns:
        List[Dict[str, Any]]: List of search results.
    """
    logger.info(f"Executing Tavily search for query: {query[:50]}...")
    return tool.invoke({"query": query})


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Web search based on the re-phrased question using Tavily API.

    Args:
        state (Dict[str, Any]): The current graph state.

    Returns:
        Dict[str, Any]: Updated state with web search results.
    """
    logger.info("Executing web search")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    progress_placeholder = st.empty()
    progress_placeholder.info("Initiating web search...")

    try:
        if not st.session_state.tavily_api_key:
            progress_placeholder.warning("Tavily API key not provided - skipping web search")
            return {"keys": {"documents": documents, "question": question}}

        progress_placeholder.info("Configuring search tool...")
        tool = TavilySearchResults(
            api_key=st.session_state.tavily_api_key,
            max_results=3,
            search_depth="advanced",
        )

        progress_placeholder.info("Executing search query...")
        try:
            search_results = execute_tavily_search(tool, question)
        except Exception as search_error:
            logger.error(f"Search failed: {str(search_error)}")
            progress_placeholder.error(f"Search failed after retries: {str(search_error)}")
            return {"keys": {"documents": documents, "question": question}}

        if not search_results:
            progress_placeholder.warning("No search results found")
            return {"keys": {"documents": documents, "question": question}}

        progress_placeholder.info("Processing search results...")
        web_results = []
        for result in search_results:
            content = (
                f"Title: {result.get('title', 'No title')}\n"
                f"Content: {result.get('content', 'No content')}\n"
            )
            web_results.append(content)

        web_document = Document(
            page_content="\n\n".join(web_results),
            metadata={
                "source": "tavily_search",
                "query": question,
                "result_count": len(web_results),
            },
        )
        documents.append(web_document)
        progress_placeholder.success(f"Successfully added {len(web_results)} search results")
        logger.info(f"Added {len(web_results)} web search results")

    except Exception as error:
        error_msg = f"Web search error: {str(error)}"
        logger.error(error_msg)
        progress_placeholder.error(error_msg)
    finally:
        progress_placeholder.empty()

    return {"keys": {"documents": documents, "question": question}}


def load_documents(file_or_url: str, is_url: bool = True) -> List[Document]:
    """Load documents from URL or file.

    Args:
        file_or_url (str): The file path or URL.
        is_url (bool): Whether the input is a URL.

    Returns:
        List[Document]: Loaded documents.
    """
    try:
        logger.info(f"Loading document from: {file_or_url}")
        if is_url:
            loader = WebBaseLoader(file_or_url)
            loader.requests_per_second = 1
        else:
            file_extension = os.path.splitext(file_or_url)[1].lower()
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_or_url)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        return loader.load()
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        st.error(f"Error loading document: {str(e)}")
        return []


st.subheader("Document Input")
input_option = st.radio("Choose input method:", ["URL", "File Upload"])

docs = None

if input_option == "URL":
    url = st.text_input("Enter document URL:", value=st.session_state.doc_url)
    if url:
        docs = load_documents(url, is_url=True)
else:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
        os.unlink(tmp_file.name)

if docs:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)

    collection_name = "rag-qdrant"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    vectorstore.add_documents(all_splits)
    retriever = vectorstore.as_retriever()
    logger.info(f"Loaded {len(all_splits)} documents into vector store")


class GraphState(TypedDict):
    """Graph state for LangGraph workflow."""

    keys: Dict[str, Any]


def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant documents from vector store.

    Args:
        state (Dict[str, Any]): Current graph state.

    Returns:
        Dict[str, Any]: Updated state with retrieved documents.
    """
    logger.info("Retrieving documents")
    state_dict = state["keys"]
    question = state_dict["question"]

    if retriever is None:
        logger.warning("Retriever not initialized")
        return {"keys": {"documents": [], "question": question}}

    documents = retriever.get_relevant_documents(question)
    logger.info(f"Retrieved {len(documents)} documents")
    return {"keys": {"documents": documents, "question": question}}


def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer using Claude 3.5 Sonnet model.

    Args:
        state (Dict[str, Any]): Current graph state.

    Returns:
        Dict[str, Any]: Updated state with generated answer.
    """
    logger.info("Generating response")
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]

    try:
        prompt = PromptTemplate(
            template="""Based on the following context, please answer the question.
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"],
        )
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=st.session_state.anthropic_api_key,
            temperature=0,
            max_tokens=1000,
        )
        context = "\n\n".join(doc.page_content for doc in documents)

        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )

        generation = rag_chain.invoke({})
        logger.info("Response generated successfully")

        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation,
            }
        }

    except Exception as e:
        error_msg = f"Error in generate function: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": "Sorry, I encountered an error while generating the response.",
            }
        }


def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Determine relevance of retrieved documents.

    Args:
        state (Dict[str, Any]): Current graph state.

    Returns:
        Dict[str, Any]: Updated state with graded documents.
    """
    logger.info("Grading document relevance")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=st.session_state.anthropic_api_key,
        temperature=0,
        max_tokens=1000,
    )

    prompt = PromptTemplate(
        template="""You are grading the relevance of a retrieved document to a user question.
        Return ONLY a JSON object with a "score" field that is either "yes" or "no".
        Do not include any other text or explanation.

        Document: {context}
        Question: {question}

        Rules:
        - Check for related keywords or semantic meaning
        - Use lenient grading to only filter clear mismatches
        - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm | StrOutputParser()

    filtered_docs = []
    search = "No"

    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            json_match = re.search(r"\{.*\}", response)
            if json_match:
                response = json_match.group()

            score = json.loads(response)

            if score.get("score") == "yes":
                logger.info("Document relevant")
                filtered_docs.append(d)
            else:
                logger.info("Document not relevant, will perform web search")
                search = "Yes"

        except Exception as e:
            logger.warning(f"Error grading document: {str(e)}")
            filtered_docs.append(d)
            continue

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}


def transform_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Transform the query to produce a better question.

    Args:
        state (Dict[str, Any]): Current graph state.

    Returns:
        Dict[str, Any]: Updated state with improved question.
    """
    logger.info("Transforming query")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = PromptTemplate(
        template="""Generate a search-optimized version of this question by
        analyzing its core semantic meaning and intent.
        \n ------- \n
        {question}
        \n ------- \n
        Return only the improved question with no additional text:""",
        input_variables=["question"],
    )

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=st.session_state.anthropic_api_key,
        temperature=0,
        max_tokens=1000,
    )

    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    logger.info(f"Query transformed: {better_question[:50]}...")

    return {"keys": {"documents": documents, "question": better_question}}


def decide_to_generate(state: Dict[str, Any]) -> str:
    """Decide whether to generate response or perform web search.

    Args:
        state (Dict[str, Any]): Current graph state.

    Returns:
        str: Next node to execute.
    """
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        logger.info("Decision: transform query and run web search")
        return "transform_query"
    else:
        logger.info("Decision: generate response")
        return "generate"


def format_document(doc: Document) -> str:
    """Format a document for display.

    Args:
        doc (Document): The document to format.

    Returns:
        str: Formatted document string.
    """
    return f"""
    Source: {doc.metadata.get('source', 'Unknown')}
    Title: {doc.metadata.get('title', 'No title')}
    Content: {doc.page_content[:200]}...
    """


def format_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format state for display.

    Args:
        state (Dict[str, Any]): The state to format.

    Returns:
        Dict[str, Any]: Formatted state.
    """
    formatted = {}

    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value

    return formatted


# Build LangGraph workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Streamlit UI
st.title("Corrective RAG Agent")
st.text("Example query: What are the experiment results and ablation studies in this research paper?")

user_question = st.text_input("Please enter your question:")

if user_question:
    logger.info(f"Processing user question: {user_question}")
    inputs = {"keys": {"question": user_question}}

    import pprint

    for output in app.stream(inputs):
        for key, value in output.items():
            with st.expander(f"Step '{key}':"):
                st.text(pprint.pformat(format_state(value["keys"]), indent=2, width=80))

    final_generation = value["keys"].get(
        "generation", "No final generation produced."
    )
    st.subheader("Final Generation:")
    st.write(final_generation)


if __name__ == "__main__":
    logger.info("Corrective RAG application started")
