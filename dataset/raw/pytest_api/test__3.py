import unittest
import requests

from tests import Server, WorkshopTest
from tests.test_workshop.GetWorkshop import GetWorkshop, GetWorkshopRepBody


class TestGetWorkshop(unittest.TestCase):

    def setUp(self):
        WorkshopTest().clean()

    def test_api_ok(self):
        """ Default case.

        Return
            200 - Workshop Get.
        """
        """ env """
        tc_workshop1 = WorkshopTest(name="qaRHR_name1").insert()
        WorkshopTest(name="qaRHR_name2").insert()
        tc_id = tc_workshop1.id
        """ call api """
        url = f'{Server.main_url}/{GetWorkshop.url}/{tc_id}'
        response = requests.get(url, verify=False)
        response_body = GetWorkshopRepBody(**response.json())
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, GetWorkshop.msg_success)
        self.assertEqual(response_body.data, GetWorkshopRepBody.data_expected(tc_workshop1))
        self.assertNotIn("detail", response_body)

    @classmethod
    def tearDownClass(cls):
        cls.setUp(TestGetWorkshop())


if __name__ == '__main__':
    unittest.main()
