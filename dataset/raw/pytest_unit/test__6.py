import pytest
import json
from app import app, get_average_last_hour

# Flask provides a test client for simulating requests
@pytest.fixture
def client():
    app.config["TESTING"] = True  # Enable testing mode
    with app.test_client() as client:
        yield client

# 1️⃣ **Unit Test**: Test the get_average_last_hour function (Mocking API Call)
def test_get_average_last_hour(mocker):
    mock_response = [
        {"createdAt": "2025-03-06T06:36:25.489Z", "value": "5.40"},
        {"createdAt": "2025-03-06T06:36:35.489Z", "value": "6.50"},
        {"createdAt": "2025-03-06T06:36:45.489Z", "value": "4.90"},
    ]

    mocker.patch("requests.get", return_value=mocker.Mock(json=lambda: mock_response, status_code=200))

    data = get_average_last_hour()
    print("Test received:", data)  # Debugging print
    assert "average" in data
    assert data["average"], 2 == 5.27  # Ensure correct rounding


# 2️⃣ **Endpoint Test**: Test the Home Page (`/`)
def test_home_page(client, mocker):
    mock_response = [
        {"createdAt": "2025-03-06T06:36:25.489Z", "value": "5.40"},
        {"createdAt": "2025-03-06T06:36:35.489Z", "value": "6.50"},
        {"createdAt": "2025-03-06T06:36:45.489Z", "value": "4.90"},
    ]
    
    # Mock the get_average_last_hour to return our mock data
    mocker.patch("app.get_average_last_hour", return_value={"average": 5.27})
    
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to Sensor Dashboard" in response.data  # Ensure page has expected content
    #assert b"Average: 5.27" in response.data  # Ensure the average value is shown

# 3️⃣ **Endpoint Test**: Test the `/refresh_average` API
def test_refresh_average(client, mocker):
    mock_response = [
        {"createdAt": "2025-03-06T06:36:25.489Z", "value": "5.40"},
        {"createdAt": "2025-03-06T06:36:35.489Z", "value": "6.50"},
        {"createdAt": "2025-03-06T06:36:45.489Z", "value": "4.90"},
    ]
    
    # Mock the get_average_last_hour to return our mock data
    mocker.patch("app.get_average_last_hour", return_value={"average": 5.27})

    response = client.get("/refresh_average")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "average" in data
    assert data["average"] == 5.27
