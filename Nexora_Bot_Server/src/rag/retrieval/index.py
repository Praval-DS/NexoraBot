from src.services.llm import openAI
from fastapi import HTTPException
from src.services.supabase import supabase
from src.rag.retrieval.utils import (
    get_project_settings,
    get_project_document_ids,
    build_context_from_retrieved_chunks,
    generate_query_variations,
)
from typing import List, Dict
from src.rag.retrieval.utils import rrf_rank_and_fuse



def _rewrite_query_for_retrieval(user_query: str) -> str:
    """
    Rewrite the user query to be more retrieval-friendly.
    Expands ambiguous terms, adds domain context, removes filler.
    Uses mini_llm — cheap call, high ROI.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a retrieval optimization expert.
Rewrite the user query to maximize semantic search recall from a document store.
Rules:
- Expand abbreviations and acronyms
- Make implicit intent explicit ("what's our policy" → "company policy on X")
- Add relevant synonyms in parentheses if helpful
- Keep it under 2 sentences
- Return ONLY the rewritten query, nothing else."""),
        ("user", "{query}")
    ])
    rewritten = (prompt | openAI["mini_llm"] | StrOutputParser()).invoke({"query": user_query})
    print(f"DEBUG: Query rewritten: '{user_query}' → '{rewritten}'")
    return rewritten.strip()




def _compress_chunks(chunks: list, user_query: str) -> list:
    """
    For each retrieved chunk, extract only the sentences relevant to the query.
    Reduces noise in the context window. Use mini_llm to keep cost low.
    Skip chunks shorter than 300 chars — not worth the LLM call.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract only the sentences from the passage that are directly relevant 
to answering the question. Keep exact wording. If the entire passage is relevant, return it as-is.
Return ONLY the extracted text, nothing else."""),
        ("user", "Question: {query}\n\nPassage:\n{passage}")
    ])
    chain = prompt | openAI["mini_llm"] | StrOutputParser()

    compressed = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if len(content) > 300:
            try:
                extracted = chain.invoke({"query": user_query, "passage": content})
                chunk = {**chunk, "content": extracted, "compressed": True}
            except Exception:
                pass  # fall back to original chunk silently
        compressed.append(chunk)

    return compressed

def retrieve_context(project_id, user_query):
    try:
        """
        RAG Retrieval Pipeline Steps:
        * Step 1: Get user's project settings from the database.
        * Step 2: Retrieve the document IDs for the current project.
        * Step 3: Perform a vector search using the RPC function to find the most relevant chunks.
        * Step 4: Perform a hybrid search (combines vector + keyword search) using RPC function.
        * Step 5: Perform multi-query vector search (generate multiple query variations and search)
        * Step 6: Perform multi-query hybrid search (multiple queries with hybrid strategy)
        * Step 7: Build the context from the retrieved chunks and format them into a structured context with citations.
        """
        # Step 1: Get user's project settings from the database.
        project_settings = get_project_settings(project_id)

        # Step 2: Retrieve the document IDs for the current project.
        document_ids = get_project_document_ids(project_id)
        # print("Found document IDs: ", len(document_ids))

        # Step 4 & 5: Execute search based on selected strategy.




        retrieval_query = _rewrite_query_for_retrieval(user_query) #query rewriting

        strategy = project_settings["rag_strategy"]
        chunks = []
        if strategy == "basic":
            # Basic RAG Strategy: Vector search only
            chunks = vector_search(retrieval_query, document_ids, project_settings)
            print(f"Vector search resulted in: {len(chunks)} chunks")

        elif strategy == "hybrid":
            # Hybrid RAG Strategy: Combines vector + keyword search with RRF ranking
            chunks = hybrid_search(retrieval_query, document_ids, project_settings)
            print(f"Hybrid search resulted in: {len(chunks)} chunks")

        # Step 6: Multi-query vector search
        elif strategy == "multi-query-vector":
            chunks = multi_query_vector_search(
                retrieval_query, document_ids, project_settings
            )
            print(f"Multi-query vector search resulted in: {len(chunks)} chunks")

        # Step 7: Multi-query hybrid search
        elif strategy == "multi-query-hybrid":
            chunks = multi_query_hybrid_search(
                retrieval_query, document_ids, project_settings
            )
            print(f"Multi-query hybrid search resulted in: {len(chunks)} chunks")

        # Step 8: Selecting top k chunks
        chunks = chunks[: project_settings["final_context_size"]]


        chunks = _compress_chunks(chunks, user_query)   # Right now retrieved chunks go directly to the LLM context as-is. Chunks often contain surrounding text irrelevant to the query so we compress chunks


        # Step 9: Build the context from the retrieved chunks and format them into a structured context with citations.
        texts, images, tables, citations = build_context_from_retrieved_chunks(chunks)
        # validate_context_from_retrieved_chunks(texts, images, tables, citations)

        return texts, images, tables, citations
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed in RAG's Retrieval: {str(e)}"
        )


def vector_search(user_query, document_ids, project_settings):
    user_query_embedding = openAI["embeddings"].embed_documents([user_query])[0]
    vector_search_result_chunks = supabase.rpc(
        "vector_search_document_chunks",
        {
            "query_embedding": user_query_embedding,
            "filter_document_ids": document_ids,
            "match_threshold": project_settings["similarity_threshold"],
            "chunks_per_search": project_settings["chunks_per_search"],
        },
    ).execute()
    return vector_search_result_chunks.data if vector_search_result_chunks.data else []


def keyword_search(query, document_ids, settings):
    keyword_search_result_chunks = supabase.rpc(
        "keyword_search_document_chunks",
        {
            "query_text": query,
            "filter_document_ids": document_ids,
            "chunks_per_search": settings["chunks_per_search"],
        },
    ).execute()

    return (
        keyword_search_result_chunks.data if keyword_search_result_chunks.data else []
    )


def hybrid_search(query: str, document_ids: List[str], settings: dict) -> List[Dict]:
    """Execute hybrid search by combining vector and keyword results"""
    # Get results from both search methods
    vector_results = vector_search(query, document_ids, settings)
    keyword_results = keyword_search(query, document_ids, settings)

    print(f"📈 Vector search returned: {len(vector_results)} chunks")
    print(f"📈 Keyword search returned: {len(keyword_results)} chunks")

    # Combine using RRF with configured weights
    return rrf_rank_and_fuse(
        [vector_results, keyword_results],
        [settings["vector_weight"], settings["keyword_weight"]],
    )


def multi_query_vector_search(user_query, document_ids, project_settings):
    """Execute multi-query vector search using query variations"""
    queries = generate_query_variations(
        user_query, project_settings["number_of_queries"]
    )
    print(f"Generated {len(queries)} query variations")

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = vector_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        print(
            f"Vector search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
        )

    final_chunks = rrf_rank_and_fuse(all_chunks)
    print(f"RRF Fusion returned {len(final_chunks)} chunks")
    return final_chunks


def multi_query_hybrid_search(user_query, document_ids, project_settings):
    """Execute multi-query hybrid search using query variations"""
    queries = generate_query_variations(
        user_query, project_settings["number_of_queries"]
    )
    print(f"Generated {len(queries)} query variations for hybrid search")

    all_chunks = []
    for index, query in enumerate(queries):
        chunks = hybrid_search(query, document_ids, project_settings)
        all_chunks.append(chunks)
        print(
            f"Hybrid search for query {index+1}/{len(queries)}: {query} resulted in: {len(chunks)} chunks"
        )

    final_chunks = rrf_rank_and_fuse(all_chunks)
    print(f"RRF Fusion returned {len(final_chunks)} chunks")
    return final_chunks
