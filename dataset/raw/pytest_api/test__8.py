import unittest
import requests

from tests import Server, WorkshopTest, WorkshopCategories
from tests.test_workshop.GetAllWorkshop import GetAllWorkshop, GetAllWorkshopRepBody


class TestGetAllWorkshop(unittest.TestCase):

    def setUp(self):
        WorkshopTest().clean()

    def test_api_ok(self):
        """ Default case.

        Return
            200 - All Workshop.
        """
        """ env """
        tc_workshop1 = WorkshopTest(name="qaRHR_name1").insert()
        tc_workshop2 = WorkshopTest(name="qaRHR_name2").insert()
        """ call api """
        url = f'{Server.main_url}/{GetAllWorkshop.url}'
        response = requests.get(url, verify=False)
        response_body = GetAllWorkshopRepBody(**response.json())
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, GetAllWorkshop.msg_success)
        self.assertIn(GetAllWorkshopRepBody.data_expected(tc_workshop1), response_body.data)
        self.assertIn(GetAllWorkshopRepBody.data_expected(tc_workshop2), response_body.data)
        self.assertNotIn("detail", response_body)

    def test_api_ok_with_args(self):
        """ With args.

        Return
            200 - All Workshop.
        """
        """ env """
        tc_workshop1 = WorkshopTest(name="qaRHR_name1", categories=WorkshopCategories.list()).insert()
        tc_workshop2 = WorkshopTest(name="qaRHR_name2", categories=[WorkshopCategories.Fitness.value]).insert()
        """ call api """
        url = f'{Server.main_url}/{GetAllWorkshop.url}?' \
              f'{GetAllWorkshop.param_categories}={WorkshopCategories.Cardio.value}'
        response = requests.get(url, verify=False)
        response_body = GetAllWorkshopRepBody(**response.json())
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, GetAllWorkshop.msg_success)
        self.assertIn(GetAllWorkshopRepBody.data_expected(tc_workshop1), response_body.data)
        self.assertNotIn(GetAllWorkshopRepBody.data_expected(tc_workshop2), response_body.data)
        self.assertNotIn("detail", response_body)

    def test_api_ok_with_args_complex(self):
        """ With args.

        Return
            200 - All Workshop.
        """
        """ env """
        tc_workshop1 = WorkshopTest(name="qaRHR_name1", description="desc1",
                                    categories=WorkshopCategories.list()).insert()
        tc_workshop2 = WorkshopTest(name="qaRHR_name2", description="desc2",
                                    categories=[WorkshopCategories.Fitness.value]).insert()
        tc_workshop3 = WorkshopTest(name="qaRHR_name3", description="desc2",
                                    categories=WorkshopCategories.list()).insert()
        tc_workshop4 = WorkshopTest(name="qaRHR_name4", description="desc1",
                                    categories=[WorkshopCategories.Fitness.value,
                                                WorkshopCategories.Cardio.value]).insert()
        """ call api """
        url = f'{Server.main_url}/{GetAllWorkshop.url}?' \
              f'{GetAllWorkshop.param_name}=qaRHR&' \
              f'{GetAllWorkshop.param_name}=name&' \
              f'{GetAllWorkshop.param_description}=desc&' \
              f'{GetAllWorkshop.param_categories}={WorkshopCategories.Strength.value}&' \
              f'{GetAllWorkshop.param_categories}={WorkshopCategories.Cardio.value}'
        response = requests.get(url, verify=False)
        response_body = GetAllWorkshopRepBody(**response.json())
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, GetAllWorkshop.msg_success)
        self.assertIn(GetAllWorkshopRepBody.data_expected(tc_workshop1), response_body.data)
        self.assertNotIn(GetAllWorkshopRepBody.data_expected(tc_workshop2), response_body.data)
        self.assertIn(GetAllWorkshopRepBody.data_expected(tc_workshop3), response_body.data)
        self.assertNotIn(GetAllWorkshopRepBody.data_expected(tc_workshop4), response_body.data)
        self.assertNotIn("detail", response_body)

    @classmethod
    def tearDownClass(cls):
        cls.setUp(TestGetAllWorkshop())


if __name__ == '__main__':
    unittest.main()
