import http from 'k6/http';
import { sleep, check } from 'k6';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

export function handleSummary(data) {
  return {
    "result.html": htmlReport(data),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}

export let options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp-up to 10 users over 30 seconds
    { duration: '1m', target: 10 },   // Stay at 10 users for 1 minute
    { duration: '10s', target: 0 },   // Ramp-down to 0 users over 10 seconds
  ],
};

export default function () {
    // Test the home page endpoint
    let res1 = http.get('https://lol.com/');
    check(res1, {
      'status is 200 for endpoint 1': (r) => r.status === 200,
      'response time for endpoint 1 is < 500ms': (r) => r.timings.duration < 500,
    });
  
    // Test the Api endpoint
    let res2 = http.get('https://lol.com/api/hello');
    check(res2, {
      'status is 200 for endpoint 2': (r) => r.status === 200,
      'response time for endpoint 2 is < 500ms': (r) => r.timings.duration < 500,
    });
  
    sleep(1); // Simulate a user waiting for 1 second before the next iteration
  }