import json
import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import re
from typing import List, Dict
import hashlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("Openai_api_key")


def extract_chapter_info(text: str, page_num: int) -> Dict[str, str]:
    """Extract chapter/section information from text."""
    # Common patterns for chapters/sections
    chapter_patterns = [
        r'^Chapter\s+(\d+|[IVXLCDM]+)[:\s]+(.+?)$',
        r'^CHAPTER\s+(\d+|[IVXLCDM]+)[:\s]+(.+?)$',
        r'^Section\s+(\d+\.?\d*)[:\s]+(.+?)$',
        r'^(\d+)\.\s+([A-Z].+?)$',
    ]
    
    lines = text.split('\n')[:5]  # Check first 5 lines
    
    for line in lines:
        line = line.strip()
        for pattern in chapter_patterns:
            match = re.match(pattern, line)
            if match:
                return {
                    "chapter_number": match.group(1),
                    "chapter_title": match.group(2).strip()
                }
    
    return {"chapter_number": "Unknown", "chapter_title": "Unknown"}


def create_parent_child_chunks(docs: List, parent_chunk_size: int = 2000, 
                               child_chunk_size: int = 500, 
                               child_overlap: int = 100) -> tuple:
    """
    Create parent-child chunks for better retrieval.
    Small children for search, large parents for context.
    """
    # Create parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parent_chunks = parent_splitter.split_documents(docs)
    
    # Create child chunks from parents
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    child_chunks = []
    parent_map = {}  # Map child to parent
    
    for parent_idx, parent in enumerate(parent_chunks):
        parent_id = hashlib.md5(
            f"{parent.page_content[:100]}{parent_idx}".encode()
        ).hexdigest()
        
        # Split parent into children
        children = child_splitter.split_documents([parent])
        
        for child_idx, child in enumerate(children):
            # Store parent info in child metadata
            child.metadata.update({
                "parent_id": parent_id,
                "parent_text": parent.page_content,
                "child_index": child_idx,
                "total_children": len(children)
            })
            child_chunks.append(child)
            parent_map[len(child_chunks) - 1] = parent_idx
    
    logger.info(f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks")
    return child_chunks, parent_chunks, parent_map


def enhance_metadata(chunks: List, pdf_path: str) -> List:
    """
    Add comprehensive metadata to chunks:
    - Page numbers
    - Chapter/section info
    - Chunk position
    - Headers/context
    """
    current_chapter = {"chapter_number": "Unknown", "chapter_title": "Unknown"}
    
    for idx, chunk in enumerate(chunks):
        # Extract page number (already in metadata from PyPDFLoader)
        page_num = chunk.metadata.get("page", 0)
        
        # Try to detect chapter info from chunk content
        chapter_info = extract_chapter_info(chunk.page_content, page_num)
        
        # Update current chapter if we found a new one
        if chapter_info["chapter_number"] != "Unknown":
            current_chapter = chapter_info
        
        # Extract potential headers (lines that are short and capitalized)
        lines = chunk.page_content.split('\n')
        headers = [
            line.strip() for line in lines[:3] 
            if len(line.strip()) < 100 and line.strip().isupper()
        ]
        
        # Enhance metadata
        chunk.metadata.update({
            "page_number": page_num + 1,  # Convert to 1-indexed
            "chapter_number": current_chapter["chapter_number"],
            "chapter_title": current_chapter["chapter_title"],
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "chunk_position": f"{idx + 1}/{len(chunks)}",
            "headers": " | ".join(headers) if headers else "",
            "source_file": os.path.basename(pdf_path)
        })
    
    return chunks


def build_vector_db(pdf_path: str, index_name: str):
    """Build a vector database from a PDF with parent-child chunking and rich metadata."""
    
    # Init Pinecone client
    pc = Pinecone(api_key=os.getenv("pinecone_api_key"))
    env = os.getenv("PINECONE_ENV")

    # Create index if not exists
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=3072,  # text-embedding-3-large dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env)
        )
        logger.info(f"✓ Created Pinecone index `{index_name}` in {env}")
    else:
        logger.info(f"✓ Using existing index `{index_name}`")

    # Load PDF
    logger.info(f"📄 Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info(f"✓ Loaded {len(docs)} pages from PDF")

    # Create parent-child chunks
    logger.info(f"✂️  Creating parent-child chunks...")
    child_chunks, parent_chunks, parent_map = create_parent_child_chunks(
        docs,
        parent_chunk_size=2000,
        child_chunk_size=500,
        child_overlap=100
    )

    # Enhance metadata
    logger.info(f"🏷️  Enhancing metadata...")
    child_chunks = enhance_metadata(child_chunks, pdf_path)

    # Use text-embedding-3-large for best quality
    logger.info(f"🔢 Generating embeddings with text-embedding-3-large...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Push to Pinecone (using child chunks for retrieval)
    logger.info(f"☁️  Uploading to Pinecone...")
    vectordb = PineconeVectorStore.from_documents(
        child_chunks,
        embeddings,
        index_name=index_name
    )

    logger.info(f"✅ Successfully stored {len(child_chunks)} child chunks in Pinecone index `{index_name}`")
    logger.info(f"✅ Each child chunk references its parent ({len(parent_chunks)} parents total)")
    
    # Print sample metadata
    if child_chunks:
        logger.info("\n📋 Sample chunk metadata:")
        sample = child_chunks[0].metadata
        for key, value in sample.items():
            if key != "parent_text":  # Don't print full parent text
                logger.info(f"   {key}: {value}")


if __name__ == "__main__":
    pdf_file = "comprehensive-clinical-nephrology.pdf"
    
    if not os.path.exists(pdf_file):
        logger.error(f"❌ PDF file not found: {pdf_file}")
        exit(1)
    
    logger.info("🚀 Building Nephrology Book Vector DB with Parent-Child Chunking...")
    build_vector_db(pdf_file, "nephrology-book")
    logger.info("🎉 Done!")