from unittest import TestCase
import time
import requests

from compose.cli.main import TopLevelCommand, project_from_options


class DockerComposeIntegrationTests(TestCase):

    def setUp(self):
        path = './'  # Path to docker-compose directory
        self.options = {
            "--no-deps": False,
            "--abort-on-container-exit": False,
            "SERVICE": "",
            "--remove-orphans": False,
            "--no-recreate": True,
            "--force-recreate": False,
            "--build": False,
            '--no-build': False,
            '--no-color': False,
            "--rmi": "none",
            "--volumes": "",
            "--follow": False,
            "--timestamps": False,
            "--tail": "all",
            "-d": True,
            '--always-recreate-deps': False,
            '--scale': []
        }

        project = project_from_options(path, self.options)
        self.compose = TopLevelCommand(project)

        self.compose.up(self.options)
        time.sleep(1)  # wait for the server to be ready

    def tearDown(self):
        self.compose.down(self.options)

    def test_compose_webservice_exists(self):
        out = requests.get('http://0.0.0.0:5000/')
        print(self.compose.project.__dict__)
        self.assertEqual(out.text, 'Hello World! I have been seen 1 times.\n')

        time.sleep(1)
        out = requests.get('http://0.0.0.0:5000/')
        self.assertEqual(out.text, 'Hello World! I have been seen 2 times.\n')
