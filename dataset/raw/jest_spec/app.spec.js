const request = require("supertest");
const mongoose = require("mongoose");
const app = require("./app");

let cookie;

beforeAll(async () => {
    const loginResponse = await request(app)
        .post("/login")
        .send("usuario=sparoj&contrasena=123")
        .expect("Content-Type", "text/plain; charset=utf-8")
        .expect(302);
    cookie = loginResponse.headers["set-cookie"][0];
});

afterAll(async () => {
    await mongoose.disconnect();
});

describe("POST /administrativo/alumno", () => {
    it("Debe almacenar un nuevo alumno con nombre, apellido, usuario y contraseña", async () => {
        const response = await request(app)
            .post("/administrativo/alumno")
            .set("Cookie", cookie)
            .send({
                nombre: "nombrePrueba",
                apellido: "apellidoPrueba",
                usuarioNombre: "usuarioPruebaNuevo",
                usuarioClave: "clavePruebe",
            })
            .expect("Content-Type", "text/html; charset=utf-8")
            .expect(201);
    });

    it("Debe retornar 409 porque el alumno ya existe", async () => {
        const response = await request(app)
            .post("/administrativo/alumno")
            .set("Cookie", cookie)
            .send({
                nombre: "xxxxx",
                apellido: "xxxxxx",
                usuarioNombre: "biondania",
                usuarioClave: "xxxxxx",
            })
            .expect("Content-Type", "text/html; charset=utf-8")
            .expect(409);
    });

    it("Debe retornar 500 porque faltan datos requeridos en el documento", async () => {
        const response = await request(app)
            .post("/administrativo/alumno")
            .set("Cookie", cookie)
            .send({
                nombre: "",
                apellido: "",
                usuarioNombre: "",
                usuarioClave: "",
            })
            .expect("Content-Type", "text/html; charset=utf-8")
            .expect(500);
    });
});

describe("POST /alumno/update/:id", () => {
    it("Debe actualizar un alumno existente", async () => {
        const response = await request(app)
            .post(`/administrativo/alumno/update/2000001`)
            .set("Cookie", cookie)
            .send({
                usuarioId: "1000001",
                _id: "2000001",
                nombre: "Daniel",
                apellido: "Biondani",
                usuarioNombre: "biondania",
                usuarioClave:
                    "$2b$04$lzXklJAf6PAfLkA9a5NEAu.ot1.OKsJrpHsMtaA03eCefBFosSkgO",
            })
            .expect("Content-Type", "text/html; charset=utf-8")
            .expect(200);
    });
});

describe("GET /alumno/delete/:id", () => {
    it("Debe eliminar un alumno existente", async () => {
        const response = await request(app)
            .get(`/administrativo/alumno/delete/2000001`)
            .set("Cookie", cookie)
            .expect(302);
    });

    it("Debe retornar 500 al intentar eliminar un alumno inexistente", async () => {
        const response = await request(app)
            .get(`/administrativo/alumno/delete/999999999`)
            .set("Cookie", cookie)
            .expect(500);
    });
});
