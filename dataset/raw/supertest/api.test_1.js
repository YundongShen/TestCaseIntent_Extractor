const request = require('supertest');
const app = require('../server');

describe('Books API', () => {

    // GET all books
    test('GET /api/books should return all books', async () => {
        const res = await request(app).get('/api/books');
        expect(res.statusCode).toBe(200);
        expect(Array.isArray(res.body)).toBe(true);
    });

    // GET by valid ID
    test('GET /api/books/:id should return a book', async () => {
        const res = await request(app).get('/api/books/1');
        expect(res.statusCode).toBe(200);
        expect(res.body.id).toBe(1);
    });

    // GET invalid ID
    test('GET /api/books/:id should return 404 for invalid ID', async () => {
        const res = await request(app).get('/api/books/999');
        expect(res.statusCode).toBe(404);
    });

    // POST new book
    test('POST /api/books should create a new book', async () => {
        const newBook = {
            title: "Dune",
            author: "Frank Herbert",
            genre: "Sci-Fi",
            copiesAvailable: 4
        };

        const res = await request(app)
            .post('/api/books')
            .send(newBook);

        expect(res.statusCode).toBe(201);
        expect(res.body.title).toBe("Dune");
    });

    // PUT update book
    test('PUT /api/books/:id should update a book', async () => {
        const res = await request(app)
            .put('/api/books/1')
            .send({ title: "Updated Title" });

        expect(res.statusCode).toBe(200);
        expect(res.body.title).toBe("Updated Title");
    });

    // PUT invalid ID
    test('PUT /api/books/:id should return 404 if not found', async () => {
        const res = await request(app)
            .put('/api/books/999')
            .send({ title: "No Book" });

        expect(res.statusCode).toBe(404);
    });

    // DELETE book
    test('DELETE /api/books/:id should delete a book', async () => {
        const res = await request(app).delete('/api/books/1');
        expect(res.statusCode).toBe(200);
    });

    // DELETE invalid ID
    test('DELETE /api/books/:id should return 404 if not found', async () => {
        const res = await request(app).delete('/api/books/999');
        expect(res.statusCode).toBe(404);
    });

});


