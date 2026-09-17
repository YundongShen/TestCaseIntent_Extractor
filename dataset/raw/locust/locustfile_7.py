from locust import HttpUser, TaskSet, task, constant
import string
import random
import time
import re
import os
import json
import traceback

class CaseInsestiveDict(dict):
    def __getitem__(self,key):
        return super(CaseInsestiveDict,self).__getitem__(key.lower())

class UserBehavior(TaskSet):
    headers= {
        "Content_Type": "application/x-www-form-urlencoded"
    }

    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        #self.comment()

    @task(52)
    def index(self):
        self.client.get("/shop")
    
    @task(44)
    def addToCart(self):
        foo = ["1", "2", "3", "4", "5", "6"] 
        num_item = random.randint(1, 10)
        data = {
            "product_sku": "",
            "product_id": f"{random.choice(foo)}", 
            "quantity": f"{num_item}"
        }
        self.res = self.client.post("/?wc-ajax=add_to_cart", data, headers=self.headers)
    
    @task(4)
    def checkout(self):
        self.addToCart()
        self.res2 = self.client.get("/checkout/", cookies=self.res.cookies)
        self.res2_nonce_list = re.findall("\"woocommerce-process-checkout-nonce\" value=\"\w+", self.res2.text)
        self.res2_nonce = self.res2_nonce_list[0]
        personal_data = {
            "billing_last_name": "○○",
            "billing_first_name": "○○",
            "billing_company": "",
            "billing_country": "JP",
            "billing_postcode": "123-4567",
            "billing_state": "JP13",
            "billing_city": "a",
            "billing_address_1": "11",
            "billing_address_2": "",
            "billing_phone": "1123456789",
            "billing_email": "xxxx@xxx.jp",
            "shipping_last_name": "○○",
            "shipping_first_name": "○○",
            "shipping_company": "",
            "shipping_country": "JP",
            "shipping_postcode": "123-4567",
            "shipping_state": "JP13",
            "shipping_city": "a",
            "shipping_address_1": "11",
            "shipping_address_2": "",
            "order_comments": "", 
            "shipping_method[0]": "free_shipping:1",
            "payment_method": "cod",
            "mailpoet_woocommerce_checkout_optin_present": "1",
            "woocommerce-process-checkout-nonce": f"{self.res2_nonce[-10:]}",
            "wp_http_referer": "http://xx.x.xx.xxx/checkout/"
        }
        self.client.post("/?wc-ajax=checkout", personal_data, cookies=self.res.cookies)
class WebsiteUser(HttpUser):
    host = 'http://x.x.xx.x/'
    tasks = {UserBehavior}
    wait_time = constant(0)
