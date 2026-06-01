import unittest
import requests

from tests import Server, WorkshopTest, WorkshopCategories
from tests.test_workshop.UpdateWorkshop import UpdateWorkshop, UpdateWorkshopRepBody


class TestUpdateWorkshop(unittest.TestCase):

    def setUp(self):
        WorkshopTest().clean()

    def test_api_ok(self):
        """ Default case.

        Return
            200 - Workshop Updated.
        """
        """ env """
        tc_workshop = WorkshopTest().insert()
        tc_id = tc_workshop.id
        body = {"_id": "qaRHR_id",
                UpdateWorkshop.param_name: "qaRHR_name_update",
                UpdateWorkshop.param_description: "qaRHR_description_update",
                UpdateWorkshop.param_categories: [WorkshopCategories.Cardio.value],
                "files": ["invalid"],
                "invalid": "invalid"}
        """ call api """
        url = f'{Server.main_url}/{UpdateWorkshop.url}/{tc_id}'
        response = requests.put(url, json=body, verify=False)
        tc_workshop_updated: WorkshopTest = UpdateWorkshop().workshop_update_from_body(workshop=tc_workshop, body=body)
        response_body = UpdateWorkshopRepBody(**response.json())
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, UpdateWorkshop.msg_success)
        self.assertEqual(response_body.data, UpdateWorkshopRepBody.data_expected(tc_workshop_updated))
        self.assertNotIn("detail", response_body)
        """ check bdd """
        tc_workshop.check_data_by_id()

    @classmethod
    def tearDownClass(cls):
        cls.setUp(TestUpdateWorkshop())


if __name__ == '__main__':
    unittest.main()
