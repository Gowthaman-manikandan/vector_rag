import json
import urllib.request
import urllib.error
import os
from pathlib import Path

from rag import (
    LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL,
    DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    answer, agentic_answer, generate_content, detect_content_type,
)
from email_tool import send_email
from jira_tool import create_ticket
from notion_tool import create_notion_page, create_notion_ticket

AGENT_CONFIG = {
  "agent_name": "EnterpriseAssistant",
  "routing_strategy": "tool_selection",
  "tools": [
    {
      "name": "vector_search",
      "description": "Search uploaded PDF documents, employee lists, sales reports, SOP documents, company policies and knowledge base."
    },
    {
      "name": "llm_generate",
      "description": "Generate new content (emails, greeting messages, marketing copy, social media captions, design descriptions, templates, etc.) when no relevant uploaded documents exist for the query."
    },
    {
      "name": "jira_tool",
      "description": "Create, update, assign and retrieve Jira tickets."
    },
    {
      "name": "email_tool",
      "description": "Send emails and retrieve email information."
    },
    {
      "name": "notion_tool",
      "description": "Create a plain Notion page with summary notes."
    },
    {
      "name": "notion_ticket_tool",
      "description": "Create a Notion database ticket/card with Title, Assignee, Status, Priority, and Description fields."
    }
  ]
}

def call_llm(prompt: str) -> str:
    """A generic LLM caller without the strict RAG context prompts."""
    provider = LLM_PROVIDER.lower()
    
    if provider == "ollama":
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"
            
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "Error: DEEPSEEK_API_KEY is not set. Add it to .env"
            
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return ""
        except Exception as e:
            return f"Error: {e}"
    
    return "Unsupported provider."

def determine_tool(query: str, history: str = "") -> str:
    prompt = f"""
You are an intelligent routing agent.
Your job is to select the NEXT logical tool to fulfill the user's request.

Available tools:
- "vector_search": Search the database for lists, policies, knowledge, or any factual info from uploaded documents.
- "llm_generate": Generate new content such as emails, greeting card messages, marketing copy, social media captions, design descriptions, or custom templates. Use this when no documents are available OR when the user explicitly asks to CREATE or GENERATE content.
- "email_tool": Send an email.
- "jira_tool": Create a Jira ticket.
- "notion_tool": Create a plain Notion page or note with general content.
- "notion_ticket_tool": Create a Notion DATABASE CARD/TICKET with Assignee, Status, Priority fields — use when the user says 'assign', 'create ticket in notion', 'add card', 'notion task'.
- "FINAL_ANSWER": The request is completely fulfilled and we are done.

Actions taken so far:
{history if history else "None"}

User Request: "{query}"

Reply ONLY with the exact name of the tool to use (e.g. 'vector_search', 'llm_generate', 'email_tool', 'jira_tool', 'notion_tool', 'notion_ticket_tool', or 'FINAL_ANSWER'). No extra text.
"""
    response = call_llm(prompt)
    for tool in ["notion_ticket_tool", "llm_generate", "vector_search", "email_tool", "jira_tool", "notion_tool", "FINAL_ANSWER"]:
        if tool in response:
            return tool
    return "FINAL_ANSWER"

def execute_agent(query: str) -> list[dict]:
    history = []
    actions_taken = ""
    
    for step in range(4):  # Max 4 steps to allow for llm_generate after vector_search
        tool = determine_tool(query, actions_taken)
        
        if tool == "vector_search":
            # Use agentic_answer which intelligently routes between RAG/hybrid/generation
            result = agentic_answer(query)
            ans = result["answer"]
            mode = result.get("mode", "rag")
            decision = result.get("decision", "")
            content_type = result.get("content_type", "general")

            # If no relevant docs found, automatically escalate to llm_generate
            if mode == "generated":
                history.append({
                    "tool": "llm_generate",
                    "output": ans,
                    "mode": mode,
                    "content_type": content_type,
                    "decision": decision,
                    "sources": result.get("sources", []),
                })
                actions_taken += f"Used llm_generate (auto-escalated from vector_search, no relevant docs). Result summary: {ans[:200]}\n"
            else:
                history.append({
                    "tool": "vector_search",
                    "output": ans,
                    "mode": mode,
                    "content_type": content_type,
                    "decision": decision,
                    "sources": result.get("sources", []),
                })
                actions_taken += f"Used vector_search (mode={mode}). Result: {ans[:200]}\n"
            
        elif tool == "llm_generate":
            content_type = detect_content_type(query)
            prompt = f"""
Generate high-quality {content_type} content based on this user request.
User Request: {query}
Additional context from prior steps: {actions_taken if actions_taken else 'None'}

Provide a complete, professional, ready-to-use response.
"""
            generated = call_llm(prompt)
            history.append({
                "tool": "llm_generate",
                "output": generated,
                "mode": "generated",
                "content_type": content_type,
                "decision": "Directly called LLM content generation tool.",
                "sources": [],
            })
            actions_taken += f"Used llm_generate to create {content_type} content.\n"

        elif tool == "email_tool":
            prompt = f"""
Generate JSON to send an email based on this user request and the data we found.
User Request: {query}
Data found so far: {actions_taken}

Return ONLY raw valid JSON format (no markdown codeblocks or other text). Example:
{{"to": ["email1@test.com", "email2@test.com"], "subject": "Greeting", "body": "Hello team!"}}
"""
            json_str = call_llm(prompt)
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # Find the first { and last } in case the LLM outputs extra text
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
            
            try:
                data = json.loads(json_str)
                res = send_email(data.get("to", []), data.get("subject", ""), data.get("body", ""))
                history.append({"tool": "email_tool", "output": res, "details": data})
                actions_taken += f"Used email_tool to send email to {data.get('to')}.\n"
            except Exception as e:
                history.append({"tool": "email_tool", "output": f"Failed to generate valid email JSON: {e}"})
                actions_taken += "Attempted to use email_tool but failed.\n"
                
        elif tool == "jira_tool":
            prompt = f"""
Generate JSON to create a Jira ticket based on this user request and the data we found.
User Request: {query}
Data found so far: {actions_taken}

Return ONLY raw valid JSON format (no markdown formatting). Example:
{{"summary": "Ticket title", "description": "Detailed description"}}
"""
            json_str = call_llm(prompt)
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # Find the first { and last }
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
                
            try:
                data = json.loads(json_str)
                res = create_ticket(data.get("summary", ""), data.get("description", ""))
                history.append({"tool": "jira_tool", "output": res, "details": data})
                actions_taken += f"Used jira_tool to create ticket.\n"
            except Exception as e:
                history.append({"tool": "jira_tool", "output": f"Failed to generate valid Jira JSON: {e}"})
                actions_taken += "Attempted to use jira_tool but failed.\n"
                
        elif tool == "notion_tool":
            prompt = f"""
Generate JSON to create a Notion page based on this user request and the data we found.
User Request: {query}
Data found so far: {actions_taken}

Return ONLY raw valid JSON format (no markdown formatting). Example:
{{"title": "Page Title", "content": "Detailed content for the page."}}
"""
            json_str = call_llm(prompt)
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
            try:
                data = json.loads(json_str)
                res = create_notion_page(data.get("title", "Untitled"), data.get("content", ""))
                history.append({"tool": "notion_tool", "output": res, "details": data})
                actions_taken += f"Used notion_tool to create page.\n"
            except Exception as e:
                history.append({"tool": "notion_tool", "output": f"Failed to generate valid Notion JSON: {e}"})
                actions_taken += "Attempted to use notion_tool but failed.\n"

        elif tool == "notion_ticket_tool":
            prompt = f"""
Generate JSON to create a Notion ticket/card in a database based on this user request.
User Request: {query}
Data found so far: {actions_taken}

Return ONLY raw valid JSON. Example:
{{"title": "Fix login bug", "description": "Users cannot log in with SSO.", "assignee_name": "John Smith", "status": "To Do", "priority": "High"}}

Allowed status values: To Do, In Progress, Done
Allowed priority values: Low, Medium, High
"""
            json_str = call_llm(prompt)
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
            try:
                data = json.loads(json_str)
                res = create_notion_ticket(
                    title=data.get("title", "Untitled Task"),
                    description=data.get("description", ""),
                    assignee_name=data.get("assignee_name", ""),
                    status=data.get("status", "To Do"),
                    priority=data.get("priority", "Medium"),
                )
                history.append({"tool": "notion_ticket_tool", "output": res, "details": data})
                actions_taken += f"Used notion_ticket_tool to create ticket '{data.get('title')}' assigned to '{data.get('assignee_name')}'.\n"
            except Exception as e:
                history.append({"tool": "notion_ticket_tool", "output": f"Failed to generate valid Notion ticket JSON: {e}"})
                actions_taken += "Attempted to use notion_ticket_tool but failed.\n"
                
        elif tool == "FINAL_ANSWER":
            break
            
    if not history:
        # Fallback to standard RAG if no tools were used
        ans = answer(query)['answer']
        history.append({"tool": "vector_search", "output": ans})
    elif "FINAL_ANSWER" in tool and history:
        # Generate summary
        prompt = f"Summarize what we did for the user based on these actions: {actions_taken}"
        final_ans = call_llm(prompt)
        history.append({"tool": "FINAL_ANSWER", "output": final_ans})
    
    return history
