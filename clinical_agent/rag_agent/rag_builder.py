import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import StateGraph, START, END
from state import State
from schema import AnswerWithSources
from prompt import retriever_content
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('Openai_api_key')
# os.environ["OPENAI_API_KEY"] = st.secrets["Openai_api_key"]
os.environ["PINECONE_API_KEY"] = os.getenv("pinecone_api_key")
os.environ["PINECONE_ENV"] = os.getenv("PINECONE_ENV")
# os.environ["OPENAI_API_KEY"] = st.secrets["Openai_api_key"]
# os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
# os.environ["PINECONE_ENV"] = st.secrets["PINECONE_ENV"]


class RAGAgent:
    def __init__(self, index_name: str, model: str = "gpt-4o-mini"):
        self.index_name = index_name
        self.model = model

        # Embeddings and LLM
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(temperature=0.7, model_name=self.model)

        # Connect to Pinecone
        self.pc = Pinecone(api_key=os.getenv("pinecone_api_key"))
        # self.pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
        # Attach to an existing Pinecone index
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embedder
        )

    #Retrieval 
    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])

        logger.info(f"Retrieved {len(retrieved_docs)} documents for the query.")
        return {"context": retrieved_docs}

    #Generation
    def generate(self, state: State):
        sources = []
        for doc in state["context"]:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source_url = doc.metadata["source"]
                if source_url not in sources:
                    sources.append(source_url)
                    print(f"Source URL: {source_url}")

        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        formatted_prompt = retriever_content.format(
            context=docs_content,
            question=state["question"],
            source=sources
        )

        logger.info(f"Formatted generation prompt: {formatted_prompt}")
        structured_llm = self.llm.with_structured_output(AnswerWithSources)
        result = structured_llm.invoke(formatted_prompt)

        return {"answer": result, "sources": sources}

    #Build Graph
    def build(self):
        subgraph = StateGraph(State)
        subgraph.add_node("retrieve", self.retrieve)
        subgraph.add_node("generate", self.generate)
        subgraph.add_edge(START, "retrieve")
        subgraph.add_edge("retrieve", "generate")
        subgraph.add_edge("generate", END)
        return subgraph.compile()