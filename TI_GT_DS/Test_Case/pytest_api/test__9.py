import unittest
import requests

from tests import Server, File, WorkshopTest
from tests.test_workshop.PostWorkshopFiles import PostWorkshopFiles, PostWorkshopFilesRepBody


class TestPostWorkshopFiles(unittest.TestCase):

    def setUp(self):
        WorkshopTest().clean()

    def test_api_ok(self):
        """ Default case.

        Return
            201 - Files Workshop Added.
        """
        """ env """
        tc_workshop = WorkshopTest().insert()
        tc_id = tc_workshop.id
        tc_file1 = File(name="qaRHR_text.txt", parent=f'workshop/{tc_id}')
        tc_file2 = File(name="qaRHR_image.png", parent=f'workshop/{tc_id}')
        tc_files = [('files', open(tc_file1.origin_path, 'rb')),
                    ('files', open(tc_file2.origin_path, 'rb'))]
        """ call api """
        url = f'{Server.main_url}/{PostWorkshopFiles.url1}/{tc_id}/{PostWorkshopFiles.url2}'
        response = requests.post(url, files=tc_files, verify=False)
        response_body = PostWorkshopFilesRepBody(**response.json())
        tc_workshop.add_files(files=[tc_file1, tc_file2])
        """ assert """
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 201)
        self.assertEqual(response_body.msg, PostWorkshopFiles.msg_success)
        self.assertEqual(response_body.data, PostWorkshopFilesRepBody.data_expected(tc_workshop))
        self.assertEqual(response_body.detail,
                         PostWorkshopFilesRepBody.detail_expected_success(_id=tc_id, files=[tc_file1, tc_file2]))
        """ check bdd """
        tc_workshop.check_data_by_id()
        """ check storage """
        tc_file1.check_file_storage(exist=True)
        tc_file2.check_file_storage(exist=True)

    @classmethod
    def tearDownClass(cls):
        cls.setUp(TestPostWorkshopFiles())
        Server.clean_tests_files()


if __name__ == '__main__':
    unittest.main()
