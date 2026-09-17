import requests
import jsonschema
import unittest
from unittest.mock import Mock

# Пример API запроса
def make_api_request():
    response = requests.get('https://api.example.com/data')
    return response.json()

# Пример теста API
class APITestCase(unittest.TestCase):
    
    def test_api_response(self):
        # Мокирование API запроса
        requests.get = Mock()
        requests.get.return_value.status_code = 200
        requests.get.return_value.json.return_value = {
            'data': {
                'name': 'John',
                'age': 30
            }
        }
        
        # Вызов функции для получения данных API
        api_data = make_api_request()
        
        # Проверка кода ответа HTTP
        self.assertEqual(requests.get.return_value.status_code, 200)
        
        # Определение схемы данных API
        schema = {
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'age': {'type': 'integer'}
                    },
                    'required': ['name', 'age']
                }
            },
            'required': ['data']
        }
        
        # Валидация данных API с использованием схемы
        jsonschema.validate(api_data, schema)
        
        # Сравнение полученных данных с ожидаемыми значениями
        expected_data = {
            'data': {
                'name': 'John',
                'age': 30
            }
        }
        self.assertEqual(api_data, expected_data)

if __name__ == '__main__':
    unittest.main()
