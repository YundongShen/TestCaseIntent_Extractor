import http from 'k6/http';
import { check, sleep } from 'k6';

// --- Configuration ---
const targetUrl = __ENV.TARGET_URL || 'http://test.k6.io';
const httpMethod = __ENV.HTTP_METHOD || 'GET';
const requestBody = __ENV.BODY || null;
const headersJson = __ENV.HEADERS_JSON || '{}';
const stagesJson = __ENV.STAGES_JSON || '[]';
const vus = __ENV.VUS ? parseInt(__ENV.VUS, 10) : undefined;
const duration = __ENV.DURATION || undefined;

// --- Parse configuration ---
let headers = {};
try {
  headers = JSON.parse(headersJson);
} catch (e) {
  console.error('Could not parse HEADERS_JSON:', e);
}

let stages = [];
try {
  stages = JSON.parse(stagesJson);
} catch (e) {
  console.error('Could not parse STAGES_JSON:', e);
}

// Set a default content-type for POST/PUT/PATCH if a body exists and it's not already set
if (requestBody && ['POST', 'PUT', 'PATCH'].includes(httpMethod.toUpperCase())) {
  if (!Object.keys(headers).some((h) => h.toLowerCase() === 'content-type')) {
    headers['Content-Type'] = 'application/json';
  }
}

// --- k6 Options ---
export const options = {
  // Ramping VUs configuration
  stages: stages.length > 0 ? stages : undefined,
  // Fixed VUs and duration configuration (used if stages is not defined)
  vus: stages.length === 0 ? vus : undefined,
  duration: stages.length === 0 ? duration : undefined,

  // Thresholds can be used to set pass/fail criteria for the test
  thresholds: {
    http_req_failed: ['rate<0.01'], // http errors should be less than 1%
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
  },
};

// --- Test Execution ---
export default function () {
  const params = {
    headers: headers,
  };

  const res = http.request(httpMethod, targetUrl, requestBody, params);

  check(res, {
    'status is 2xx': (r) => r.status >= 200 && r.status < 300,
  });

  // A small sleep to simulate user think time
  sleep(1);
}
