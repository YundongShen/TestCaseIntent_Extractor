import unittest
import requests

from tests import Server, File, WorkshopTest
from tests.test_workshop.DeleteWorkshopFiles import DeleteWorkshopFiles, DeleteWorkshopFilesRepBody


class TestDeleteWorkshopFiles(unittest.TestCase):

    def setUp(self):
        WorkshopTest().clean()

    def test_api_ok(self):
        """ Default case.

        Return
            200 - Files Workshop Delete.
        """
        """ env """
        tc_workshop = WorkshopTest().insert()
        tc_id = tc_workshop.id
        tc_file1 = File(name="qaRHR_text.txt", parent=f'workshop/{tc_id}').create()
        tc_file2 = File(name="qaRHR_image.png", parent=f'workshop/{tc_id}').create()
        tc_workshop.add_files_in_mongo(files=[tc_file1, tc_file2])
        """ call api """
        url = f'{Server.main_url}/{DeleteWorkshopFiles.url1}/{tc_id}/{DeleteWorkshopFiles.url2}?' \
              f'{DeleteWorkshopFiles.param_filenames}[]={tc_file1.name}&' \
              f'invalid=invalid'
        response = requests.delete(url, verify=False)
        response_body = DeleteWorkshopFilesRepBody(**response.json())
        tc_workshop.delete_files(files=[tc_file1])
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, DeleteWorkshopFiles.msg_success)
        self.assertEqual(response_body.data, DeleteWorkshopFilesRepBody.data_expected(tc_workshop))
        self.assertEqual(response_body.detail,
                         DeleteWorkshopFilesRepBody.detail_expected_success(files=[tc_file1]))
        """ check bdd """
        tc_workshop.check_data_by_id()
        """ check storage """
        tc_file1.check_file_storage(exist=False)
        tc_file2.check_file_storage(exist=True)

    def test_api_ok_multi(self):
        """ Default case multi files.

        Return
            200 - Files Workshop Delete.
        """
        """ env """
        tc_workshop = WorkshopTest().insert()
        tc_id = tc_workshop.id
        tc_file1 = File(name="qaRHR_text.txt", parent=f'workshop/{tc_id}').create()
        tc_file2 = File(name="qaRHR_image.png", parent=f'workshop/{tc_id}').create()
        tc_workshop.add_files_in_mongo(files=[tc_file1, tc_file2])
        """ call api """
        url = f'{Server.main_url}/{DeleteWorkshopFiles.url1}/{tc_id}/{DeleteWorkshopFiles.url2}?' \
              f'{DeleteWorkshopFiles.param_filenames}[]={tc_file1.name}&' \
              f'{DeleteWorkshopFiles.param_filenames}[]={tc_file2.name}'
        response = requests.delete(url, verify=False)
        response_body = DeleteWorkshopFilesRepBody(**response.json())
        tc_workshop.delete_files(files=[tc_file1, tc_file2])
        """ assert """
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response_body.status, 200)
        self.assertEqual(response_body.msg, DeleteWorkshopFiles.msg_success)
        self.assertEqual(response_body.data, DeleteWorkshopFilesRepBody.data_expected(tc_workshop))
        self.assertEqual(response_body.detail,
                         DeleteWorkshopFilesRepBody.detail_expected_success(files=[tc_file1, tc_file2]))
        """ check bdd """
        tc_workshop.check_data_by_id()
        """ check storage """
        tc_file1.check_file_storage(exist=False)
        tc_file2.check_file_storage(exist=False)

    @classmethod
    def tearDownClass(cls):
        cls.setUp(TestDeleteWorkshopFiles())
        Server.clean_tests_files()


if __name__ == '__main__':
    unittest.main()
