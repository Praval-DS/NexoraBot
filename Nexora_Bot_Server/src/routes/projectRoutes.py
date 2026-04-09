from typing import Dict, List
import time
from fastapi import APIRouter, HTTPException, Depends
from src.agents.simple_agent.agent import create_simple_rag_agent
from src.agents.supervisor_agent.agent import create_supervisor_agent
import httpx
from src.services.llm import openAI
import json

from src.services.supabase import supabase
from src.services.clerkAuth import get_current_user_clerk_id
from src.models.index import ProjectCreate, ProjectSettings
from src.models.index import MessageCreate, MessageRole
import os
import shutil
from src.services.awsS3 import s3_client
from src.config.index import appConfig
from src.agents.csv_agent import create_project_csv_agent


router = APIRouter(tags=["projectRoutes"])
"""
`/api/projects`

  - GET `/api/projects/` ~ List all projects
  - POST `/api/projects/` ~ Create a new project
  - DELETE `/api/projects/{project_id}` ~ Delete a specific project
  
  - GET `/api/projects/{project_id}` ~ Get specific project data
  - GET `/api/projects/{project_id}/chats` ~ Get specific project chats
  - GET `/api/projects/{project_id}/settings` ~ Get specific project settings
  
  - PUT `/api/projects/{project_id}/settings` ~ Update specific project settings
  - POST `/api/projects/{project_id}/chats/{chat_id}/messages` ~ Send a message to a Specific Chat
  
"""

@router.get("/")
async def get_projects(current_user_clerk_id: str = Depends(get_current_user_clerk_id)):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Query projects table for projects related to the current user
    * 3. Return projects data
    """
    start_time = time.time()
    print(f"[Profiling] get_projects start: {start_time}")

    try:
        projects_query_result = (
            supabase.table("projects")
            .select("*")
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        db_time = time.time()
        print(f"[Profiling] Database query took: {db_time - start_time}s")


        return {
            "message": "Projects retrieved successfully",
            "data": projects_query_result.data or [],
        }

    except HTTPException as e:
        raise e

    except httpx.ConnectError as e:
        print(f"Connection Error in get_projects: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Could not connect to database. Please ensure Supabase is running.",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching projects: {str(e)}",
        )


@router.post("/")
async def create_project(
    project_data: ProjectCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Insert new project into database
    * 3. Check if project creation failed, then return error
    * 4. Create default project settings for the new project
    * 5. Check if project settings creation failed, then rollback the project creation
    * 6. Return newly created project data
    """
    try:
        # Insert new project into database
        project_insert_data = {
            "name": project_data.name,
            "description": project_data.description,
            "clerk_id": current_user_clerk_id,
        }

        project_creation_result = (
            supabase.table("projects").insert(project_insert_data).execute()
        )

        if not project_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create project - invalid data provided",
            )

        newly_created_project = project_creation_result.data[0]

        # Create default project settings for the new project
        project_settings_data = {
            "project_id": newly_created_project["id"],
            "embedding_model": "text-embedding-3-large",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "reranker-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }

        project_settings_creation_result = (
            supabase.table("project_settings").insert(project_settings_data).execute()
        )

        if not project_settings_creation_result.data:
            # Rollback: Delete the project if settings creation fails
            supabase.table("projects").delete().eq(
                "id", newly_created_project["id"]
            ).execute()
            raise HTTPException(
                status_code=422,
                detail="Failed to create project settings - project creation rolled back",
            )

        return {
            "message": "Project created successfully",
            "data": newly_created_project,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while creating project: {str(e)}",
        )


@router.delete("/{project_id}")
async def delete_project(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Verify if the project exists and belongs to the current user
    * 3. Delete project - CASCADE will automatically delete all related data:
    * 4. Check if project deletion failed, then return error
    * 5. Return successfully deleted project data
    """
    try:
        # Verify if the project exists and belongs to the current user
        project_ownership_verification_result = (
            supabase.table("projects")
            .select("id")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not project_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,  # Not Found - project doesn't exist or doesn't belong to user
                detail="Project not found or you don't have permission to delete it",
            )

        # Delete project ~ "CASCADE" will automatically delete all related data: project_settings, project_documents, document_chunks, chats, messages, etc.
        project_deletion_result = (
            supabase.table("projects")
            .delete()
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not project_deletion_result.data:
            raise HTTPException(
                status_code=500,  # Internal Server Error - deletion failed unexpectedly
                detail="Failed to delete project - please try again",
            )

        successfully_deleted_project = project_deletion_result.data[0]

        return {
            "message": "Project deleted successfully",
            "data": successfully_deleted_project,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while deleting project: {str(e)}",
        )


@router.get("/{project_id}")
async def get_project(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Verify if the project exists and belongs to the current user
    * 3. Return project data
    """
    try:
        project_result = (
            supabase.table("projects")
            .select("*")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not project_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to access it",
            )

        return {
            "message": "Project retrieved successfully",
            "data": project_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project: {str(e)}",
        )


@router.get("/{project_id}/chats")
async def get_project_chats(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Verify if the project exists and belongs to the current user
    * 3. Return project chats data
    """
    try:
        project_chats_result = (
            supabase.table("chats")
            .select("*")
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .order("created_at", desc=True)
            .execute()
        )

        # * If there are no chats for the project, return an empty list
        # * A User may or may not have any chats for a project

        return {
            "message": "Project chats retrieved successfully",
            "data": project_chats_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} chats: {str(e)}",
        )


@router.get("/{project_id}/settings")
async def get_project_settings(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Verify if the project exists and belongs to the current user
    * 3. Check if the project settings exists for the project
    * 4. Return project settings data
    """
    try:
        project_settings_result = (
            supabase.table("project_settings")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )

        if not project_settings_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project settings not found or you don't have permission to access it",
            )

        return {
            "message": "Project settings retrieved successfully",
            "data": project_settings_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} settings: {str(e)}",
        )


@router.put("/{project_id}/settings")
async def update_project_settings(
    project_id: str,
    settings: ProjectSettings,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Verify if the project exists and belongs to the current user
    * 3. Verify if the project settings exist for the project
    * 4. Update project settings
    * 5. Check if project settings update failed, then return error
    * 6. Return successfully updated project settings data
    """
    try:
        project_ownership_verification_result = (
            supabase.table("projects")
            .select("id")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not project_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to update its settings",
            )

        project_settings_ownership_verification_result = (
            supabase.table("project_settings")
            .select("id")
            .eq("project_id", project_id)
            .execute()
        )

        if not project_settings_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project settings not found for this project",
            )

        project_settings_update_data = (
            settings.model_dump()  # Pydantic modal to dictionary conversion
        )
        project_settings_update_result = (
            supabase.table("project_settings")
            .update(project_settings_update_data)
            .eq("project_id", project_id)
            .execute()
        )

        if not project_settings_update_result.data:
            raise HTTPException(
                status_code=422, detail="Failed to update project settings"
            )

        return {
            "message": "Project settings updated successfully",
            "data": project_settings_update_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while updating project {project_id} settings: {str(e)}",
        )

def get_chat_history(chat_id: str, exclude_message_id: str = None) -> List[Dict[str, str]]:
    """
    Fetch and format chat history for agent context.
    
    Retrieves the last 10 messages (5 user + 5 assistant) from the chat,
    excluding the current message being processed.
    
    Args:
        chat_id: The ID of the chat
        exclude_message_id: Optional message ID to exclude from history
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    try:
        query = (
            supabase.table("messages")
            .select("id, role, content")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
        )
        
        # Exclude current message if provided
        if exclude_message_id:
            query = query.neq("id", exclude_message_id)
        
        messages_result = query.execute()
        
        if not messages_result.data:
            return []
        
        # Get last 10 messages (limit to 10 total messages)
        recent_messages = messages_result.data[-10:]
        
        # Format messages for agent
        formatted_history = []
        for msg in recent_messages:
            formatted_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        return formatted_history
    except Exception:
        # If history retrieval fails, return empty list
        return []



# @router.post("/{project_id}/chats/{chat_id}/messages")
# async def send_message(
#     project_id: str,
#     chat_id: str,
#     message: MessageCreate,
#     current_user_clerk_id: str = Depends(get_current_user_clerk_id),
# ):
#     """
#     Step 1 : Insert the message into the database.
#     Step 2 : Check for structured vs unstructured files.
#     Step 3 : Pipeline A (Structured) - Invoke CSV Agent if CSV/Excel files exist.
#     Step 4 : Pipeline B (Unstructured) - Invoke RAG Agent if other files exist or fallback.
#     Step 5 : Combine responses and insert into database.
#     """
#     try:
#         # Step 1 : Insert the message into the database.
#         message_content = message.content
#         message_insert_data = {
#             "content": message_content,
#             "chat_id": chat_id,
#             "clerk_id": current_user_clerk_id,
#             "role": MessageRole.USER.value,
#         }
#         message_creation_result = (
#             supabase.table("messages").insert(message_insert_data).execute()
#         )
#         if not message_creation_result.data:
#             raise HTTPException(status_code=422, detail="Failed to create message")
        
#         current_message_id = message_creation_result.data[0]["id"]
        
#         # Step 2 : Analyze available files to determine pipelines
#         project_docs_result = supabase.table("project_documents").select("*").eq("project_id", project_id).eq("processing_status", "completed").execute()
#         project_docs = project_docs_result.data or []
        
#         structured_files = []
#         unstructured_files = []
#         print("=== PIPELINE DEBUG START ===")

#         print(f"Structured files count: {len(structured_files)}")
#         print(f"Unstructured files count: {len(unstructured_files)}")
        
#         for doc in project_docs:
#             fname = doc.get("filename", "").lower()
#             if fname.endswith(('.csv', '.xlsx', '.xls')):
#                 structured_files.append(doc)
#             else:
#                 unstructured_files.append(doc)
        
#         csv_response_text = ""
#         rag_response_text = ""
#         citations = [] # RAG only mostly

#         # Step 3 : Structured Pipeline (CSV Agent)
#         if structured_files:
#             print(">>> SQL PIPELINE TRIGGERED")

#             # We need both structured files AND a schema definition to run the smart agent
#             schema_path = None
            
#             # Check for schema file (any .json file) in project documents
#             for doc in project_docs:
#                 if doc.get("filename", "").lower().endswith(".json"):
#                     s3_key = doc["s3_key"]
#                     fname = doc["filename"]
#                     temp_dir = f"/tmp/schema_agent/{project_id}"
#                     os.makedirs(temp_dir, exist_ok=True)
#                     local_path = os.path.join(temp_dir, fname)
                    
#                     # Download schema
#                     try:
#                         s3_client.download_file(appConfig["s3_bucket_name"], s3_key, local_path)
#                         schema_path = local_path
#                     except Exception as e:
#                         print(f"Failed to download schema: {e}")
#                     break

#             if schema_path:
#                 try:
#                     # Create temp dir for this project's CSVs
#                     temp_dir = f"/tmp/csv_agent/{project_id}"
#                     os.makedirs(temp_dir, exist_ok=True)
                    
#                     local_csv_paths = []
#                     for doc in structured_files:
#                         s3_key = doc["s3_key"]
#                         fname = doc["filename"]
#                         local_path = os.path.join(temp_dir, fname)
                        
#                         # Download file (overwrite or check existence - downloading ensures freshness)
#                         s3_client.download_file(appConfig["s3_bucket_name"], s3_key, local_path)
#                         local_csv_paths.append(local_path)
                    
#                     # Invoke NEW Smart Agent
#                     from src.agents.smart_sql_agent import create_smart_agent
                    
#                     agent = create_smart_agent(local_csv_paths, schema_path)
#                     result = agent.execute_and_answer(message_content)
                    
#                     if isinstance(result, dict):
#                          csv_response_text = result.get("answer", "No answer generated.")
#                     else:
#                          csv_response_text = str(result)
                    
#                 except Exception as e:
#                     print(f"Smart SQL Agent failed: {e}")
#                     csv_response_text = f"[Error analyzing structured data: {str(e)}]"
#             else:
#                  csv_response_text = "[Notice: Structured files found but no JSON schema file was detected. Please upload a .json schema file to process the data.]"

#         # Step 4 : Unstructured Pipeline (RAG Agent / Web Search)
#         # We run this pipeline to handle:
#         # 1. Unstructured documents (PDFs, etc.)
#         # 2. Web Search (if Agentic mode)
#         # 3. General conversation
#         should_run_rag = True
        
#         if should_run_rag:
#             print(">>> RAG PIPELINE TRIGGERED")

#             # Step 2 (orig) : Get project settings
#             try:
#                 project_settings = await get_project_settings(project_id)
#                 agent_type = project_settings["data"].get("agent_type", "simple")
#             except Exception:
#                 agent_type = "simple"
                
#             # Step 3 (orig) : Get chat history
#             chat_history = get_chat_history(chat_id, exclude_message_id=current_message_id)
            
#             agent = None
#             if agent_type == "simple":
#                 agent = create_simple_rag_agent(
#                     project_id=project_id,
#                     chat_history=chat_history
#                 )
#             elif agent_type == "agentic":
#                 agent = create_supervisor_agent(
#                     project_id=project_id,
#                     chat_history=chat_history
#                 )

#             if agent:
#                 result = agent.invoke({
#                     "messages": [{"role": "user", "content": message_content}]
#                 })
#                 rag_response_text = result["messages"][-1].content
#                 citations = result.get("citations", [])

#         # Step 5 : Combine Responses
#         final_response = ""
        
#         # Check if RAG response is just a generic "I don't know" or similar, and if we have a valid CSV response
#         # Check if RAG response is just a generic "I don't know" or similar, and if we have a valid CSV response
#         rag_is_generic = False
#         if rag_response_text:
#              lower_rag = rag_response_text.lower()
#              generic_phrases = [
#                  "project documents do not contain",
#                  "i searched available resources but found no",
#                  "information not present",
#                  "to find this information, you would typically",
#                  "you would typically need to query",
#                  "no relevant chunks found",
#                  "analysis from structured data",
#                  "executed sql"
#              ]
#              if any(phrase in lower_rag for phrase in generic_phrases):
#                  rag_is_generic = True
#                  rag_response_text = "No relevant chunks found"

#         if csv_response_text and rag_response_text:
#             final_response = f"**Analysis from Structured Data:**\n{csv_response_text}\n\n**Analysis from Documents:**\n{rag_response_text}"
#         elif csv_response_text:
#             final_response = csv_response_text
#         elif rag_response_text:
#             final_response = rag_response_text
#         else:
#             final_response = "I searched available resources but found no relevant information or experienced an error."

#         # Insert AI Response
#         ai_response_insert_data = {
#             "content": final_response,
#             "chat_id": chat_id,
#             "clerk_id": current_user_clerk_id,
#             "role": MessageRole.ASSISTANT.value,
#             "citations": citations,
#         }

#         ai_response_creation_result = (
#             supabase.table("messages").insert(ai_response_insert_data).execute()
#         )
#         if not ai_response_creation_result.data:
#             raise HTTPException(status_code=422, detail="Failed to create AI response")

#         return {
#             "message": "Message created successfully",
#             "data": {
#                 "userMessage": message_creation_result.data[0],
#                 "aiMessage": ai_response_creation_result.data[0],
#             },
#         }

#     except HTTPException as e:
#         raise e

#     except Exception as e:
#         import traceback
#         traceback.print_exc()  # prints full stack trace to terminal
#         raise HTTPException(
#             status_code=500,
#             detail=f"An internal server error occurred while creating message: {str(e)}",
#         )

def classify_query_intent_llm(query: str) -> str:
    prompt = f"""
Classify the user query into one of:
- structured → SQL/data analysis
- unstructured → document understanding
- both → requires both

Return ONLY JSON:
{{"intent": "structured" | "unstructured" | "both"}}

Query: {query}
"""

    try:
        res = openAI["mini_llm"].invoke(prompt)
        parsed = json.loads(res.content)
        intent = parsed.get("intent", "both")

        if intent not in ["structured", "unstructured", "both"]:
            return "both"

        return intent

    except Exception as e:
        print(f"Intent classification failed: {e}")
        return "both"

@router.post("/{project_id}/chats/{chat_id}/messages")
async def send_message(
    project_id: str,
    chat_id: str,
    message: MessageCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    try:
        # Step 1 : Insert user message
        message_content = message.content

        message_insert_data = {
            "content": message_content,
            "chat_id": chat_id,
            "clerk_id": current_user_clerk_id,
            "role": MessageRole.USER.value,
        }

        message_creation_result = (
            supabase.table("messages").insert(message_insert_data).execute()
        )

        if not message_creation_result.data:
            raise HTTPException(status_code=422, detail="Failed to create message")

        current_message_id = message_creation_result.data[0]["id"]

        # Step 2 : Fetch project documents
        project_docs_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("project_id", project_id)
            .eq("processing_status", "completed")
            .execute()
        )

        project_docs = project_docs_result.data or []

        structured_files = []
        unstructured_files = []

        for doc in project_docs:
            fname = doc.get("filename", "").lower()

            if fname.endswith((".csv", ".xlsx", ".xls")):
                structured_files.append(doc)
            else:
                unstructured_files.append(doc)

        print("=== PIPELINE DEBUG START ===")
        print(f"Structured files: {[d.get('filename') for d in structured_files]}")
        print(f"Unstructured files: {[d.get('filename') for d in unstructured_files]}")
        print(f"Structured count: {len(structured_files)}")
        print(f"Unstructured count: {len(unstructured_files)}")



        intent = classify_query_intent_llm(message_content)
        print(f"Query intent (LLM): {intent}")

        run_sql = len(structured_files) > 0 and intent in ["structured", "both"]
        run_rag = len(unstructured_files) > 0 and intent in ["unstructured", "both"]
        structured_response_text = ""
        rag_response_text = ""
        citations = []

        # Step 3 : Structured Pipeline (SQL Agent)
        if run_sql:
            print(">>> SQL PIPELINE TRIGGERED")

            schema_path = None

            # Find schema file
            for doc in project_docs:
                fname = doc.get("filename", "").lower()

                if fname.endswith(".json"):
                    s3_key = doc["s3_key"]
                    temp_dir = f"/tmp/schema_agent/{project_id}"
                    os.makedirs(temp_dir, exist_ok=True)

                    local_path = os.path.join(temp_dir, doc["filename"])

                    try:
                        s3_client.download_file(
                            appConfig["s3_bucket_name"],
                            s3_key,
                            local_path,
                        )
                        schema_path = local_path
                    except Exception as e:
                        print(f"Schema download failed: {e}")
                    break

            if schema_path:
                try:
                    temp_dir = f"/tmp/csv_agent/{project_id}"
                    os.makedirs(temp_dir, exist_ok=True)

                    local_csv_paths = []

                    for doc in structured_files:
                        s3_key = doc["s3_key"]
                        local_path = os.path.join(temp_dir, doc["filename"])

                        s3_client.download_file(
                            appConfig["s3_bucket_name"],
                            s3_key,
                            local_path,
                        )

                        local_csv_paths.append(local_path)

                    from src.agents.smart_sql_agent import create_smart_agent

                    agent = create_smart_agent(local_csv_paths, schema_path)

                    result = agent.execute_and_answer(message_content)

                    if isinstance(result, dict):
                        structured_response_text = result.get("answer", "")
                    else:
                        structured_response_text = str(result)

                except Exception as e:
                    print(f"Smart SQL Agent failed: {e}")
                    structured_response_text = f"[SQL Error: {str(e)}]"

            else:
                structured_response_text = (
                    "[Schema missing: upload .json schema to analyze structured data]"
                )

        # Step 4 : RAG Pipeline

        if run_rag:
            print(">>> RAG PIPELINE TRIGGERED")

            try:
                project_settings = await get_project_settings(project_id)
                agent_type = project_settings["data"].get("agent_type", "simple")
            except Exception:
                agent_type = "simple"

            chat_history = get_chat_history(
                chat_id,
                exclude_message_id=current_message_id,
            )

            if agent_type == "simple":
                agent = create_simple_rag_agent(
                    project_id=project_id,
                    chat_history=chat_history,
                )
            else:
                agent = create_supervisor_agent(
                    project_id=project_id,
                    chat_history=chat_history,
                )

            if agent:
                result = agent.invoke({
                    "messages": [{"role": "user", "content": message_content}]
                })

                rag_response_text = result["messages"][-1].content
                citations = result.get("citations", [])

        print("=== PIPELINE DEBUG END ===")

        # Step 5 : Merge response
        if structured_response_text and rag_response_text:
            final_response = (
                f"**Structured Data:**\n{structured_response_text}\n\n"
                f"**Documents:**\n{rag_response_text}"
            )
        elif structured_response_text:
            final_response = structured_response_text
        elif rag_response_text:
            final_response = rag_response_text
        else:
            final_response = "No relevant information found."

        # Step 6 : Save AI response
        ai_response = {
            "content": final_response,
            "chat_id": chat_id,
            "clerk_id": current_user_clerk_id,
            "role": MessageRole.ASSISTANT.value,
            "citations": citations,
        }

        ai_result = supabase.table("messages").insert(ai_response).execute()

        if not ai_result.data:
            raise HTTPException(status_code=422, detail="Failed to create AI response")

        return {
            "message": "Message created successfully",
            "data": {
                "userMessage": message_creation_result.data[0],
                "aiMessage": ai_result.data[0],
            },
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}",
        )