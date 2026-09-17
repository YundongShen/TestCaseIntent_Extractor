from locust import HttpUser, task, between


class WebUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def index_page(self):
        self.client.get(url="/")

    @task
    def predict_page(self):
        self.client.post(
            url="/predict",
            data={
                "hotel": "City Hotel",
                "market_segment": "Online TA",
                "deposit_type": "No Deposit",
                "lead_time": 1,
                "days_in_waiting_list": 1,
                "previous_cancellations": 1,
                "previous_bookings_not_canceled": 1,
                "booking_changes": 1,
                "total_of_special_requests": 1,
            },
        )
