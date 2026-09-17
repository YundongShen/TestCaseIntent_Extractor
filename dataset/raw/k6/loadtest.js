// @ts-check
import http from 'k6/http'
import { check, fail, sleep, group } from 'k6'
import { Trend } from 'k6/metrics'

let GetPostsTrend = new Trend('Get posts', true)
let CreatePostTrend = new Trend('Create post', true)
let CreateCommentTrend = new Trend('Create comment', true)
let LikePostTrend = new Trend('Like post', true)
let ViewPostTrend = new Trend('View post', true)

export let options = {
  vus: 40,
  duration: '10s',
}

/**
   500: http_req_failed................: 30.97% ✓ 902       ✗ 2010   
   14.51% 22.52%

       ✗ status 200 (get posts)
        ↳  16% — ✓ 83 / ✗ 418
       ✗ status 200 (create post)
        ↳  31% — ✓ 155 / ✗ 336
       ✗ status 200 (create comment)
        ↳  75% — ✓ 116 / ✗ 38
       ✗ status 200 (view post)
        ↳  63% — ✓ 96 / ✗ 56
       ✗ status 200 (like post)
        ↳  60% — ✓ 84 / ✗ 54

  250: http_req_failed................: 11.72% ✓ 513        ✗ 3863 

       ✗ status 200 (get posts)
        ↳  66% — ✓ 337 / ✗ 172
       ✗ status 200 (create post)
        ↳  76% — ✓ 390 / ✗ 119
       ✗ status 200 (create comment)
        ↳  85% — ✓ 334 / ✗ 56
       ✗ status 200 (view post)
        ↳  81% — ✓ 318 / ✗ 72
       ✗ status 200 (like post)
        ↳  75% — ✓ 296 / ✗ 94

  ???
  100: http_req_failed................: 43.61% ✓ 560       ✗ 724  

       ✗ status 200 (get posts)
        ↳  6% — ✓ 17 / ✗ 259
       ✗ status 200 (create post)
        ↳  10% — ✓ 30 / ✗ 246
       ✗ status 200 (create comment)
        ↳  70% — ✓ 21 / ✗ 9
       ✗ status 200 (view post)
        ↳  33% — ✓ 10 / ✗ 20
       ✗ status 200 (like post)
        ↳  13% — ✓ 4 / ✗ 26

  50: http_req_failed................: 31.30% ✓ 273       ✗ 599 

         ✗ status 200 (get posts)
        ↳  20% — ✓ 29 / ✗ 114
       ✗ status 200 (create post)
        ↳  34% — ✓ 50 / ✗ 93
       ✗ status 200 (create comment)
        ↳  84% — ✓ 42 / ✗ 8
       ✗ status 200 (view post)
        ↳  50% — ✓ 25 / ✗ 25
       ✗ status 200 (like post)
        ↳  34% — ✓ 17 / ✗ 33

  5: 
 */

const SLEEP_DURATION = 0.1

// const baseUrl = 'http://localhost:3000/api'
const baseUrl = `https://${__ENV.API_URL}/api`

// Get posts          http localhost:3000/api/posts
// Get post/comments  http localhost:3000/api/posts/1
// Create post        http POST localhost:3000/api/posts
// Create comment     http POST localhost:3000/api/comment postId=1
// View post          http PUT localhost:3000/api/post/1/views
// Like post          http PUT localhost:3000/api/post/1/likes

// eslint-disable-next-line import/no-anonymous-default-export
export default function () {
  group('user flow', function () {
    // Get posts
    let getPostsRes = http.get(`${baseUrl}/posts`)
    check(getPostsRes, { 'status 200 (get posts)': (r) => r.status == 200 })
    GetPostsTrend.add(getPostsRes.timings.duration)

    sleep(SLEEP_DURATION)

    // Create Post
    let createPostRes = http.post(`${baseUrl}/posts`)
    if (
      !check(createPostRes, {
        'status 200 (create post)': (r) => r.status == 200,
      })
    ) {
      fail('Create post failed!')
    }
    CreatePostTrend.add(createPostRes.timings.duration)
    const createdPostId = JSON.parse(String(createPostRes.body)).id

    sleep(SLEEP_DURATION)

    // Create comment
    let createCommentRes = http.post(
      `${baseUrl}/comments`,
      JSON.stringify({
        postId: createdPostId,
        comment: 'comment from loadtest',
      }),
    )
    check(createCommentRes, {
      'status 200 (create comment)': (r) => r.status == 200,
    })
    CreateCommentTrend.add(createCommentRes.timings.duration)

    sleep(SLEEP_DURATION)

    // Add view to post
    let createViewRes = http.put(`${baseUrl}/posts/${createdPostId}/views`)
    check(createViewRes, {
      'status 200 (view post)': (r) => r.status == 200,
    })
    ViewPostTrend.add(createViewRes.timings.duration)

    sleep(SLEEP_DURATION)

    // Add view to post
    let createLikeRes = http.put(`${baseUrl}/posts/${createdPostId}/likes`)
    check(createLikeRes, {
      'status 200 (like post)': (r) => r.status == 200,
    })
    LikePostTrend.add(createLikeRes.timings.duration)
  })
}


/**
 * vus = 1000
 * 
 * 
 * █ user flow
    ✗ status 200 (get posts)
    ↳  87% — ✓ 365 / ✗ 53
    ✗ status 200 (create post)
    ↳  97% — ✓ 319 / ✗ 7
    ✗ status 200 (create comment)
    ↳  98% — ✓ 286 / ✗ 3
    ✗ status 200 (view post)
    ↳  99% — ✓ 217 / ✗ 2
    ✗ status 200 (like post)
    ↳  97% — ✓ 192 / ✗ 4 
 */