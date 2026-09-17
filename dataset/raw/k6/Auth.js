import http from 'k6/http';
import { check } from 'k6';
import { Smoke } from "./options.js";

export let options = Smoke;

// export let options = {
//     vus: 1,            // 1 virtual user
//     iterations: 20,    // 20 requests in total
// };

export default function () {
    let url = 'https://dummyjson.com/auth/login';
    let payload = JSON.stringify({
        "username": "kminchelle",
        "password": "0lelplR"
    });

    let params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    let res = http.post(url, payload, params);
    console.log(res.body);
    // Check if the status code returned is 200 (OK)
    check(res, {
        'status was 200': (r) => r.status == 200,
    });

    sleep(1);
}
