const request = require("supertest");
const app = require("./app");
describe("", () => {
  test("should return response 200 and helloo", done => {
    request(app)
      .get("/")
      .expect("Helllooooo")
      .expect(200, done);
  });

  test("should return wm ", done => {
    request(app)
      .get("/watermelon")
      .expect("watermelon 🐓")
      .expect(200, done);
  });
  test("should return wm", async () => {
    await request(app)
      .get("/watermelon")
      .expect(200)
      .expect("watermelon 🐓");
  });
});
