# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from src.config.index import appConfig

# openAI = {
#     "embeddings_llm": ChatOpenAI(
#         model="gpt-4o", api_key=appConfig["openai_api_key"], temperature=0
#     ),
#     "embeddings": OpenAIEmbeddings(
#         model="text-embedding-3-large",
#         api_key=appConfig["openai_api_key"],
#         dimensions=1536,  # ! Do not changes this value. It is used in the document_chunks embedding vector.
#     ),
#     "chat_llm": ChatOpenAI(
#         model="gpt-4o", api_key=appConfig["openai_api_key"], temperature=0
#     ),
#     "mini_llm": ChatOpenAI(
#         model="gpt-4o-mini", api_key=appConfig["openai_api_key"], temperature=0
#     ),
# }

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config.index import appConfig

openAI = {
    # Used for chunk summarization during ingestion (has vision for images)
    "embeddings_llm": AzureChatOpenAI(
        azure_deployment=appConfig["azure_chat_deployment"],
        azure_endpoint=appConfig["azure_openai_base_url"],
        api_key=appConfig["azure_openai_api_key"],
        api_version=appConfig["azure_openai_api_version"],
        temperature=0
    ),
    # Used for RAG answers and LLM responses
    "chat_llm": AzureChatOpenAI(
        azure_deployment=appConfig["azure_chat_deployment"],
        azure_endpoint=appConfig["azure_openai_base_url"],
        api_key=appConfig["azure_openai_api_key"],
        api_version=appConfig["azure_openai_api_version"],
        temperature=0
    ),
    # Used for guardrails and query variations
    "mini_llm": AzureChatOpenAI(
        azure_deployment=appConfig["azure_chat_deployment"],
        azure_endpoint=appConfig["azure_openai_base_url"],
        api_key=appConfig["azure_openai_api_key"],
        api_version=appConfig["azure_openai_api_version"],
        temperature=0
    ),
    # Used for generating vector embeddings
    "embeddings": AzureOpenAIEmbeddings(
        azure_deployment=appConfig["azure_embedding_deployment"],
        azure_endpoint=appConfig["azure_openai_base_url"],
        api_key=appConfig["azure_openai_api_key"],
        api_version=appConfig["azure_openai_api_version"]
    ),
}