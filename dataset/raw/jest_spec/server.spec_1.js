const request = require("supertest");
const server = require("./server.js");

describe("server.js index route", () => {
  it("should return 200 status code", async () => {
    const response = await request(server).get("/");

    expect(response.status).toEqual(200);
  });

  it("should return a response in text format", async () => {
    const response = await request(server).get("/");

    expect(response.type).toEqual("text/html");
  });

  it("should return a string with expected content", async () => {
    const expected = "Hello City";

    const response = await request(server).get("/");

    expect(response.text).toMatch(expected);
  });
});
