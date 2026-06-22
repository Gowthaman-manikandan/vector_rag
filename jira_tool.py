import os
import json
import urllib.request
import urllib.error
import base64

def create_ticket(summary: str, description: str) -> str:
    jira_url = os.getenv("JIRA_URL") # e.g., https://yourdomain.atlassian.net
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_project_key = os.getenv("JIRA_PROJECT_KEY") # e.g., IT

    if not all([jira_url, jira_email, jira_api_token, jira_project_key]) or jira_url == "https://yourcompany.atlassian.net":
        return "Error: Jira settings are not configured properly in .env"

    url = f"{jira_url.rstrip('/')}/rest/api/3/issue"
    
    auth_str = f"{jira_email}:{jira_api_token}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")

    payload = {
        "fields": {
            "project": {"key": jira_project_key},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }
                ]
            },
            "issuetype": {"name": "Task"} # Assuming standard issuetype 'Task'
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            ticket_key = data.get("key", "UNKNOWN")
            return f"Successfully created Jira ticket {ticket_key}."
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore").strip()
        return f"Jira API error HTTP {exc.code}: {details}"
    except Exception as e:
        return f"Failed to create Jira ticket: {e}"
