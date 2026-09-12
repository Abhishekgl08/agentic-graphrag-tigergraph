import requests

from app.tigergraph import create_connection


def test_tigergraph_schema():
    conn = create_connection()

    conn.getToken()
    token = conn.apiToken

    url = f"{conn.host}/gsql/v1/schema"

    response = requests.get(
        url,
        params={"graph": conn.graphname},
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        verify=False,
        timeout=30,
    )

    print("\n=== TigerGraph Schema ===")
    print("Status:", response.status_code)
    print(response.text)

    assert response.status_code == 200, (
        f"Schema request failed: "
        f"{response.status_code} {response.text}"
    )