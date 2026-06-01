
import os
import sys

# Add the src directory to the path
#os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "danelfin_demo"))
)
import pytest
from pydantic import EmailStr
from tinydb import TinyDB
from loguru import logger
from module_method.functions import CustomerManager

# Mocking TinyDB instance for testing
@pytest.fixture
def mock_db(tmpdir):
    db_path = os.path.join(os.path.dirname(__file__), "danelfin_demo/dbtest.json")
    print(db_path)
    db = TinyDB(db_path)
    return db

# Test case for CustomerManager class
class TestCustomerManager:
    def test_add_customer(self, mock_db):
        customer_manager = CustomerManager()
        customer_info = customer_manager.add_customer(name='John Doe', email='john@example.com', age=30, country='USA')

        assert 'name' in customer_info
        assert 'email' in customer_info
        assert 'age' in customer_info
        assert 'country' in customer_info
        assert 'id' in customer_info
   
if __name__ == '__main__':
    import pytest
    pytest.main(['-v'])
