import http from "k6/http";
import {check, sleep} from "k6";
import {Counter} from "k6/metrics";

// A simple counter for http requests

export const requests = new Counter("http_reqs");

// you can specify stages of your test (ramp up/down patterns) through the options object
// target is the number of VUs you are aiming for

export const options = {
    stages: [
        {target: 40, duration: "1m"},
        {target: 20, duration: "1m"},
        {target: 0, duration: "1m"},
    ],
    thresholds: {
        requests: ["count < 1200"],
    },
};

export default function () {
    // our HTTP request, note that we are saving the response to res, which can be accessed later
    const payload = JSON.stringify({
        dna: ["ATGCGA", "CAGTGC", "TTATGT", "AGAAGG", "CCCCTA", "TCACTG"]
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };


    const res = http.post("http://simian-api.us-east-1.elasticbeanstalk.com/dna/simian", payload, params);

    sleep(1);

    const checkRes = check(res, {
        "status is 200": (r) => r.status === 200,
    });
}
