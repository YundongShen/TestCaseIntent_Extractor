const request = require('supertest');
// Import MongoDB module
const { ObjectId } = require('mongodb');

// import json web token
const jwt = require('jsonwebtoken');

// secret key
const secret = 'tHI$_iSz_a_Very_$trong&_SeCRet_queYZ_fOR_Team32';

// get the token
const jwtoken = jwt.sign({ username: 'testUser' }, secret, { expiresIn: '60000s' });

const { closeMongoDBConnection, connect } = require('./dbFunctions');
const webapp = require('./server');

let mongo;

// TEST PUT ENDPOINT
describe('Update a comment endpoint integration test', () => {
  /**
 * If you get an error with afterEach
 * inside .eslintrc.json in the
 * "env" key add -'jest': true-
 */

  let res;
  let db;
  let testCommentID; // will store the id of the test comment

  /**
     * Make sure that the data is in the DB before running
     * any test
     * connect to the DB
     */
  beforeAll(async () => {
    mongo = await connect();
    db = mongo.db();
    // post a testUser
    await db.collection('users').insertOne({ name: 'testUser', username: 'testUser' });
    res = await request(webapp).post('/comment')
      .set({ Authorization: `${jwtoken}` })
      .send('name=testuser&userId=123&postId=456&content=how are you');
    // get the id of the test comment
    // eslint-disable-next-line no-underscore-dangle
    testCommentID = JSON.parse(res.text).data._id;
  });

  const clearDatabase = async () => {
    try {
      // delete a testUser
      await db.collection('users').deleteOne({ username: 'testUser' });
      await db.collection('comments').deleteOne({ name: 'testuser' });
    } catch (err) {
      console.log('error', err.message);
    }
  };
  /**
 * Delete all test data from the DB
 * Close all open connections
 */
  afterAll(async () => {
    await clearDatabase();
    try {
      await mongo.close();
      await closeMongoDBConnection(); // mongo client that started server.
    } catch (err) {
      return err;
    }
    return null;
  });

  test('Endpoint status code and response async/await', async () => {
    res = await request(webapp).put(`/comment/${testCommentID}`)
      .set({ Authorization: `${jwtoken}` })
      .send(`commentId=${testCommentID}&content=music is my life`);
    expect(res.status).toEqual(200);
    expect(res.type).toBe('application/json');

    // the database was updated
    const updatedComment = await db.collection('comments').findOne({ _id: ObjectId(testCommentID) });
    expect(updatedComment.content).toEqual('music is my life');
  });

  test('missing content 404', async () => {
    res = await request(webapp).put(`/comment/${testCommentID}`)
      .set({ Authorization: `${jwtoken}` })
      .send(`commentId=${testCommentID}`);
    expect(res.status).toEqual(404);
  });
});
