'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawnSync } = require('node:child_process')
const { observation, observationReady } = require('./survival-probe')

function completeBot () {
  return {
    entity: { position: { x: 0.5, y: 64, z: -1.5 } },
    health: 20,
    food: 20,
    game: { dimension: 'overworld', gameMode: 'survival' },
    inventory: { items: () => [] }
  }
}

test('incomplete health, food, or position cannot be serialized as a Survival observation', () => {
  for (const invalid of [undefined, null, NaN, Infinity, '20']) {
    for (const field of ['health', 'food', 'x', 'y', 'z']) {
      const bot = completeBot()
      if (field === 'health' || field === 'food') bot[field] = invalid
      else bot.entity.position[field] = invalid
      assert.equal(observationReady(bot), false)
      assert.throws(() => observation(bot), /not ready/)
    }
  }
  assert.equal(observationReady({}), false)
})

test('known zero health is recorded as dead rather than treated as missing', () => {
  const bot = completeBot()
  bot.health = 0
  bot.food = 0
  const result = JSON.parse(JSON.stringify(observation(bot)))
  assert.equal(result.health, 0)
  assert.equal(result.food, 0)
  assert.equal(result.dead, true)
})

function runFakeProbe (provideHealth) {
  const temporaryRoot = fs.realpathSync(os.tmpdir())
  const directory = fs.mkdtempSync(path.join(temporaryRoot, 'survival-probe-test-'))
  try {
    const preload = path.join(directory, 'fake-mineflayer.cjs')
    fs.writeFileSync(preload, `
      const Module = require('node:module')
      const { EventEmitter } = require('node:events')
      const original = Module._load
      Module._load = function (name, ...args) {
        if (name !== 'mineflayer') return original.call(this, name, ...args)
        return { createBot () {
          const bot = new EventEmitter()
          bot.entity = { position: { x: 0.5, y: 64, z: -1.5 } }
          bot.game = { dimension: 'overworld', gameMode: 'survival' }
          bot.inventory = { items: () => [] }
          bot.quit = () => bot.emit('end', 'test finished')
          queueMicrotask(() => {
            bot.emit('spawn')
            if (${provideHealth}) {
              bot.health = 20
              bot.food = 18
              bot.emit('health')
            }
          })
          return bot
        } }
      }
    `)
    return spawnSync(process.execPath, ['--require', preload, path.join(__dirname, 'survival-probe.js'), '25567', '1'], {
      encoding: 'utf8', timeout: 5000
    })
  } finally {
    const relative = path.relative(temporaryRoot, fs.realpathSync(directory))
    assert(!relative.startsWith('..') && !path.isAbsolute(relative) && relative.startsWith('survival-probe-test-'))
    fs.rmSync(directory, { recursive: true, force: true })
  }
}

test('spawn followed by health packet emits only complete observations and then completion', () => {
  const result = runFakeProbe(true)
  assert.ifError(result.error)
  assert.equal(result.status, 0, result.stderr)
  const events = result.stdout.trim().split('\n').map(line => JSON.parse(line))
  const complete = events.pop()
  assert.equal(complete.event, 'connection_check_complete')
  assert.equal(complete.observations, events.length)
  assert(events.length >= 1)
  for (const event of events) {
    assert.equal(event.health, 20)
    assert.equal(event.food, 18)
    assert.equal(event.dead, false)
    assert.equal(event.position.y, 64)
  }
})

test('spawn without complete observations times out with failure rather than a completion event', () => {
  const result = runFakeProbe(false)
  assert.ifError(result.error)
  assert.equal(result.status, 2)
  assert.equal(result.stdout, '')
  assert.match(result.stderr, /deadline passed without complete health, food, and position/)
})
