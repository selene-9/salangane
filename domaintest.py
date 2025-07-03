
SECRET_TOKEN = "0f8e1c7a-2b4d-45f3-824e-1e6a2c8b9d3e"

# Configuration
JIRA_DOMAIN = "https://something.selene9domain-test.re/rest/api/2"
API_TOKEN = "3f6n1c7a-2b4d-45f3-824e-1e6a2c8b9d3e"  # dummy token

def get_issue(issue_key):
    url = f"{JIRA_DOMAIN}/issue/{issue_key}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
