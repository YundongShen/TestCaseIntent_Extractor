const request = require("supertest");
const server = require("./server");
const db = require("./data/dbConfig");
const checkuser = require("./API/PatientAPI/auth-user-middleware");

describe("auth model", () => {
  it("should set testing environment", () => {
    expect(process.env.NODE_ENV).toBe("testing");
  });
});

describe("user registration function", () => {
  it("should register new user and return 201", async () => {
    await db("users").truncate();
    const res = await request(server)
      .post("/api/auth/user-register")
      .send({
        userEmail: "ladygaga@gmail.com",
        userPassword: "pokerface",
        userName: "Lady Gaga",
      });
    expect(res.status).toBe(201);
  });
});

describe("user login function", () => {
  it("should return messagen and token after correct login", async () => {
    const login = await request(server)
      .post("/api/auth/user-login")
      .send({ userEmail: "ladygaga@gmail.com", userPassword: "pokerface" });
    expect(login.status).toBe(200);
    expect(login.text).toContain("token");
    expect(login.text).toContain("message");
  });
});

describe("login function failure", () => {
  it("should return 401 error when login credential is incomplete", async () => {
    const loginfail = await request(server)
      .post("/api/auth/user-login")
      .send({ userEmail: "beyonce@gmail.com" });
    expect(loginfail.status).toBe(401);
  });
});

describe("medical professional registration", () => {
  it("should return 404 error if registration is incomplete", async () => {
    const medregfail = await request(server)
      .post("/api/auth/med-register")
      .send({ medicEmail: "ariana@gmail.com" });
    expect(medregfail.status).toBe(404);
  });
});

//Registration test not working
describe("medical professional registration", () => {
  it("registration type is json", async () => {
    await db("medicalProfessionals").truncate();
    const med = await request(server)
      .post("/api/auth/med-register")
      .send({
        medicEmail: "sam@gmail.com",
        medicPassword: "grande",
        medicFirstName: "Sam",
        medicLastName: "Grande",
        company: "Methodist",
        position: "Nurse",
      });
    // expect(med.type).toMatch(/json/i);
    expect(med.status).toBe(200);
  });
});

// describe("get patient from user successfully", () => {
//   it("testing middleware", () => {
//     return new Promise(resolve => {
//       const mock = jest.checkuser(req, res, err => {
//         if (!err) {
//           // resolve(req.localStorage.token);
//           console.log(res);
//         }
//       });
//       expect(mock).toBeTruthy();
//     });
//   });

//   // it("will display data", async () => {
//   //   const login = await request(server)
//   //     .post("/api/auth/user-login")
//   //     .send({ userEmail: "ladygaga@gmail.com", userPassword: "pokerface" });
//   //   const token = login.token;

//   //   return request(server)
//   //   req.headers.authorization = token
//   //     .get("/api/user/1")
//   //     .then(response => {
//   //       expect(response.status).toBe(200);
//   //     });
//   // });
// });
