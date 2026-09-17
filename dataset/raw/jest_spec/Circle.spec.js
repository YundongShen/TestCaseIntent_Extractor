/* eslint-env jest */

const Circle = require('./Circle')

describe('Support for iteration protocols', () => {
  it('Iterable: for...of', () => {
    const source = [0, 1, 2]
    const circle = new Circle(source)
    let i = 0
    const expected = [0, 1, 2]
    for (const b of circle) {
      expect(b).toEqual(expected[i++])
      if (i >= expected.length) break
    }
  })

  it('Iterator: next()', () => {
    const source = [0, 1, 2]
    const circle = new Circle(source)
    const expected = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    for (let i = 0; i < expected.length; i += 1) {
      const next = circle.next()
      expect(next.value).toEqual(expected[i])
      expect(next.done).toBe(false)
    }
  })

  it('Iterable: destruct', () => {
    const source = [0, 1, 2]
    const circle = new Circle(source)
    const [first] = circle
    expect(first).toEqual(0)
  })
})

describe('Circles elements of source array', () => {
  it('returns elements of source in a circle', () => {
    const source = [0, 1, 2]
    const circle = new Circle(source)
    expect(circle.next()).toEqual({ value: 0, done: false })
    expect(circle.next()).toEqual({ value: 1, done: false })
    expect(circle.next()).toEqual({ value: 2, done: false })
    expect(circle.next()).toEqual({ value: 0, done: false })
    expect(circle.next()).toEqual({ value: 1, done: false })
    expect(circle.next()).toEqual({ value: 2, done: false })
  })

  it('Respects size == 1', () => {
    const source = [0]
    const circle = new Circle(source)
    expect(circle.next()).toEqual({ value: 0, done: false })
    expect(circle.next()).toEqual({ value: 0, done: false })
    expect(circle.next()).toEqual({ value: 0, done: false })
  })
})

describe('Treats edge cases with respect', () => {
  it('Returns completed iterator if source size is 0', () => {
    const circle = new Circle([])
    expect(circle.next()).toEqual({ done: true })
  })

  it('Throws if source is not provided', () => {
    expect(() => new Circle()).toThrow(TypeError)
  })
})
