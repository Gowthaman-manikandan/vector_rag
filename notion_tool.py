import os
import json
import urllib.request
import urllib.error


def _notion_request(url: str, payload: dict) -> dict:
    """Helper to make a POST request to the Notion API."""
    api_key = os.getenv("NOTION_API_KEY")
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _make_bullet_blocks(items: list[str]) -> list[dict]:
    """Convert a list of strings into Notion bulleted list blocks."""
    return [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": item}}]
            }
        }
        for item in items
    ]


def _make_heading(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def _make_paragraph(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }
    }


def create_notion_page(title: str, content: str) -> str:
    """Create a simple Notion page under a parent page."""
    api_key = os.getenv("NOTION_API_KEY")
    parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID")

    if not api_key or not parent_page_id or api_key == "your_notion_api_key":
        return "Error: NOTION_API_KEY or NOTION_PARENT_PAGE_ID not configured properly in .env"

    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {
            "title": {"title": [{"text": {"content": title}}]}
        },
        "children": [_make_paragraph(content)]
    }

    try:
        data = _notion_request("https://api.notion.com/v1/pages", payload)
        page_url = data.get("url", "UNKNOWN")
        return f"Successfully created Notion page. URL: {page_url}"
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore").strip()
        return f"Notion API error HTTP {exc.code}: {details}"
    except Exception as e:
        return f"Failed to create Notion page: {e}"


def create_notion_ticket(
    title: str,
    description: str,
    assignee_name: str = "",
    status: str = "To Do",
    priority: str = "Medium",
    requirements: list[str] | None = None,
    acceptance_criteria: list[str] | None = None,
    notes: str = "",
) -> str:
    """
    Create a rich ticket card in a Notion Database with full sections.

    The database must have these properties:
      - Name      (title)
      - Status    (select):   To Do | In Progress | Done
      - Priority  (select):   Low | Medium | High
      - Assignee  (rich_text)
      - Description (rich_text)
    """
    api_key = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")

    if not api_key or not database_id or api_key == "your_notion_api_key":
        return "Error: NOTION_API_KEY or NOTION_DATABASE_ID not configured properly in .env"

    # Build rich page body
    children: list[dict] = []

    if description:
        children.append(_make_heading("📝 Description"))
        children.append(_make_paragraph(description))

    if requirements:
        children.append(_make_heading("✅ Requirements"))
        children.extend(_make_bullet_blocks(requirements))

    if acceptance_criteria:
        children.append(_make_heading("🎯 Acceptance Criteria"))
        children.extend(_make_bullet_blocks(acceptance_criteria))

    if notes:
        children.append(_make_heading("📌 Notes"))
        children.append(_make_paragraph(notes))

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {
                "title": [{"text": {"content": title}}]
            },
            "Status": {
                "select": {"name": status}
            },
            "Priority": {
                "select": {"name": priority}
            },
            "Assignee": {
                "rich_text": [{"text": {"content": assignee_name}}]
            },
            "Description": {
                "rich_text": [{"text": {"content": description[:2000]}}]  # Notion cap
            },
        },
        "children": children
    }

    try:
        data = _notion_request("https://api.notion.com/v1/pages", payload)
        page_url = data.get("url", "UNKNOWN")
        return f"Successfully created Notion ticket '{title}'. URL: {page_url}"
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore").strip()
        return f"Notion API error HTTP {exc.code}: {details}"
    except Exception as e:
        return f"Failed to create Notion ticket: {e}"
