import requests
import json
from jsonschema import validate

test_movie = {
    "actor_facets": "",
    "actors": "",
    "alternative_titles": "",
    "color": "#726B70",
    "genre": "['Action', 'Drama', 'Romance']",
    "image": "",
    "objectID": "123456",
    "rating": 5,
    "score": 6,
    "title": "test123456",
    "year": 2022,
}

test_updated_movie = {
    "actor_facets": "",
    "actors": "",
    "alternative_titles": "",
    "color": "#726B70",
    "genre": "['Action', 'Drama', 'Romance']",
    "image": "",
    "objectID": "123456",
    "rating": 5,
    "score": 6,
    "title": "test123456",
    "year": 2023,
}

schema = {
    "type": "object",
    "properties": {
        "actor_facets": {"const": ""},
        "actors": {"const": ""},
        "alternative_titles": {"const": ""},
        "color": {"const": "#726B70"},
        "genre": {"const": "['Action', 'Drama', 'Romance']"},
        "image": {"const": ""},
        "objectID": {"const": "123456"},
        "rating": {"const": 5},
        "score": {"const": 6},
        "title": {"const": "test123456"},
        "year": {"const": 2023},
    },
}


def test_get_movies_api():
    response = requests.get("http://127.0.0.1:5000/api/v1/movies")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"


def test_add_movie_api():
    headers = {"Content-Type": "application/json"}
    payload = test_movie
    response = requests.post(
        "http://127.0.0.1:5000/api/v1/movies/add",
        headers=headers,
        data=json.dumps(payload, indent=4),
    )
    assert response.status_code == 200


def test_update_movie_api():
    headers = {"Content-Type": "application/json"}
    payload = test_updated_movie
    response = requests.put(
        "http://127.0.0.1:5000/api/v1/movies/update",
        headers=headers,
        data=json.dumps(payload, indent=4),
    )
    assert response.status_code == 200


def test_get_movie_api():
    response = requests.get("http://127.0.0.1:5000/api/v1/movies/123456")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    resp_body = response.json()
    validate(instance=resp_body, schema=schema)


def test_delete_movie_api():
    response = requests.delete("http://127.0.0.1:5000/api/v1/movies/delete/123456")
    assert response.status_code == 200
