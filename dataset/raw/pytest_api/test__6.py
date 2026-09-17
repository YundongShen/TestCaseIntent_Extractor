import unittest
import requests

from tests import Server, WorkshopTest
from tests.test_workshop.DeleteWorkshop import DeleteWorkshop, DeleteWorkshopRepBody


class TestDeleteWorkshop(unittest.TestCase):

    def setUp(self):
        WorkshopTest().clean()

    def test_api_ok(self):
        """ Default case.

        Return
            200 - Workshop Deleted.
        """
        """ env """
        tc_workshop1 = WorkshopTest(name="qaRHR_name1").insert()
        tc_workshop2 = WorkshopTest(name="qaRHR_name2").insert()
        tc_id = tc_workshop1.id
        """ call api """
        url = f'{Server.main_url}/{DeleteWorkshop.url}/{tc_id}'
        response = requests.delete(url, verify=False)
        response_body = DeleteWorkshopRepBody(**response.json())
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, DeleteWorkshop.msg_success)
        self.assertEqual(response_body.data, DeleteWorkshopRepBody.data_expected(tc_workshop1))
        self.assertNotIn("detail", response_body)
        """ check bdd"""
        tc_workshop1.check_doesnt_exist_by_id()
        tc_workshop2.check_data_by_id()

    @classmethod
    def tearDownClass(cls):
        cls.setUp(TestDeleteWorkshop())


if __name__ == '__main__':
    unittest.main()
