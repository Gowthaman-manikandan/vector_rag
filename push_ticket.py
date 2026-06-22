"""
Run this script once to push the Card Login Integration ticket to Notion.
Usage: python push_ticket.py
"""
from dotenv import load_dotenv
load_dotenv()

from notion_tool import create_notion_ticket

result = create_notion_ticket(
    title="Card Login Integration",
    description="Implement the Card Login Integration module within the application and integrate it with the existing authentication system.",
    assignee_name="balaji@tendersoftware.in",
    priority="High",
    status="To Do",
    requirements=[
        "Develop the card-based login functionality according to the project requirements.",
        "Integrate the card authentication flow with the existing login system.",
        "Validate card credentials securely and ensure proper authentication.",
        "Implement comprehensive error handling and user-friendly validation messages.",
        "Ensure compatibility with the current user management and access control mechanisms.",
        "Test all login scenarios, including successful authentication, invalid card credentials, expired or inactive cards, and edge cases.",
        "Update all relevant technical and user documentation.",
    ],
    acceptance_criteria=[
        "Users can successfully log in using the card authentication method.",
        "Existing login and authentication features continue to function without issues.",
        "Appropriate validation and error messages are displayed.",
        "All test cases pass successfully.",
        "Code is reviewed, committed, and deployed to the development environment.",
        "Documentation is updated and complete.",
    ],
    notes="Please provide regular progress updates and notify the team once development, testing, and deployment have been completed.",
)

print(result)
