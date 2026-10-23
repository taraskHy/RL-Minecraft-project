'use strict'

// Dependency/import verification only. Does not create a bot or open a socket.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const { createRequire } = require('node:module')
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const requireFromMineflayer = createRequire(require.resolve('mineflayer'))
const data = requireFromMineflayer('minecraft-data')('1.21.4')

assert.equal(require('mineflayer/package.json').version, '4.39.0')
assert.equal(require('mineflayer-pathfinder/package.json').version, '2.4.5')
assert.equal(typeof pathfinder, 'function')
assert.equal(typeof Movements, 'function')
assert.equal(typeof goals.GoalNear, 'function')
assert.equal(typeof mineflayer.createBot, 'function')
assert(mineflayer.testedVersions.includes('1.21.4'))
assert.equal(new Vec3(1, 2, 3).offset(1, 0, 0).x, 2)
for (const name of ['oak_log', 'oak_planks', 'stick', 'wooden_pickaxe']) {
  assert(Number.isInteger(data.itemsByName[name]?.id), `Missing item registry entry: ${name}`)
}
for (const name of ['stone', 'air', 'barrier', 'lava', 'oak_log', 'crafting_table']) {
  assert(Number.isInteger(data.blocksByName[name]?.id), `Missing block registry entry: ${name}`)
}
const inventoryPlugin = fs.readFileSync(require.resolve('mineflayer/lib/plugins/inventory.js'), 'utf8')
assert.match(inventoryPlugin, /bot\._syncWindow = syncWindow/)
console.log('Pinned dependencies, Minecraft 1.21.4 registry, and inventory sync source contract loaded. No game connection attempted.')
