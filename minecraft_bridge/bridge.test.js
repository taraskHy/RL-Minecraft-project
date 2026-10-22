'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { parseArgs, validateStart, gridPosition, expectedArenaBlock, verifyArena } = require('./bridge')

test('world modifications require explicit development flag and local target', () => {
  assert.equal(parseArgs([]).devArena, false)
  assert.equal(parseArgs(['--dev-arena']).devArena, true)
  assert.throws(() => parseArgs(['--host', 'example.com']), /localhost/)
  assert.throws(() => parseArgs(['--username', 'bot @a']), /username/)
  assert.throws(() => parseArgs(['--origin', '0', '64']), /three integers/)
})

test('initial state cannot place the bot in hazards or occupied log cells', () => {
  assert.deepEqual(validateStart([4, 4]), [4, 4])
  for (const start of [[1, 1], [3, 2], [2, 3], [-1, 0], [5, 4], [0.5, 0]]) {
    assert.throws(() => validateStart(start))
  }
})

test('observations reject drift rather than claiming a commanded location', () => {
  assert.deepEqual(gridPosition({ x: 100.5, y: 64, z: -29.5 }, [100, 64, -30]), [0, 0])
  assert.throws(() => gridPosition({ x: 100.99, y: 64, z: -29.5 }, [100, 64, -30]), /cell center/)
  assert.throws(() => gridPosition({ x: 100.5, y: 65, z: -29.5 }, [100, 64, -30]), /cell center/)
  assert.throws(() => gridPosition({ x: 110.5, y: 64, z: -29.5 }, [100, 64, -30]), /bounds/)
  assert.throws(() => gridPosition({ x: NaN, y: 64, z: -29.5 }, [100, 64, -30]), /non-finite/)
})

test('reset verification checks containment below lava and all floor cells', () => {
  const intactArena = (x, y, z) => ({ name: expectedArenaBlock(x, y, z) })
  assert.doesNotThrow(() => verifyArena(intactArena))
  assert.equal(expectedArenaBlock(2, -1, 3), 'lava')
  assert.equal(expectedArenaBlock(2, -2, 3), 'stone')
  assert.throws(() => verifyArena((x, y, z) => x === 2 && y === -2 && z === 3 ? { name: 'air' } : intactArena(x, y, z)), /expected stone/)
  assert.throws(() => verifyArena((x, y, z) => x === 0 && y === -1 && z === 0 ? { name: 'air' } : intactArena(x, y, z)), /expected stone/)
})
