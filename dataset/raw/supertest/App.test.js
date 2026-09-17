const createServer = require("./App")
const request = require("supertest");
const app = require("./App");

describe('POST /send-message', function() {
    it('responds with json', function(done) {
      request(app)
        .post('/send-message')
        .send({name: 'john'})
        .set('Accept', 'application/json')
        .expect('Content-Type', /json/)
        .expect(400, done);
    });
  });

  describe('POST /send-message', function() {
    it('responds with json', function(done) {
      request(app)
        .post('/send-message')
        .send({"sender": "testing", 
            "registrationIds": ["b2d0e95bfce054a3d220da12280cfdbb15274a1a9e5a02e5618c72b6ee32d0e2b"], 
            "data": {"topic":"com.whatever.test","mydata":"12341234","aps":{"alert":"hello"}}
        })
        .set('Accept', 'application/json')
        .expect('Content-Type', /json/)
        .expect(200, done);
    });
  });