import random
from locust import HttpUser, task


class Task(HttpUser):
    token = ''
    user_id = 0
    group_id = 0

    @task
    def test(self):
        action = random.randint(1, 4)
        if action == 1:
            self.client.get('/api/profile', headers={'Authorization': self.token})
        if action == 2:
            self.client.get('/api/groups', headers={'Authorization': self.token})
        if action == 3:
            self.client.get(f'/api/group/{self.group_id}', headers={'Authorization': self.token})
        if action == 4:
            self.client.post('/api/split', json={"group_id": self.group_id,
                                                 "amount": 100,
                                                 "lander_id": self.user_id,
                                                 "payer_ids": [self.user_id - 1, self.user_id - 2]}, headers={'Authorization': self.token})

    def on_start(self):
        self.client.post('/api/register', json={'username': 'u' + str(random.random()), 'password': 'pass'})
        self.client.post('/api/register', json={'username': 'u' + str(random.random()), 'password': 'pass'})
        res = self.client.post('/api/register', json={'username': 'u' + str(random.random()), 'password': 'pass'}).json()
        self.token = 'Bearer ' + res['token']
        self.user_id = res['id']
        res = self.client.post('/api/group', json={'name': 'group', 'member_ids': [self.user_id, self.user_id - 1, self.user_id - 2]}, headers={'Authorization': self.token}).json()
        self.group_id = res['id']
