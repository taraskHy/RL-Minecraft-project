'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const { parseArgs, validateAction, inventoryCounts, validatePhysicalState, craftingRecipe, safeMovements, SearchMemory, replaceableForTable, ACTIONS } = require('./resource-bridge')

test('natural resource bridge only connects locally and has no reset/command interface', () => {
  assert.deepEqual(parseArgs([]), { host: '127.0.0.1', port: 25567, username: 'RLResourceBot', version: '1.21.4' })
  assert.equal(parseArgs(['--port', '25568']).port, 25568)
  for (const args of [['--host', 'example.com'], ['--port', '0'], ['--port', '2.5'], ['--username', 'bad name'], ['--version', '1.21.5'], ['--dev-arena']]) assert.throws(() => parseArgs(args))
  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  assert.doesNotMatch(source, /bot\.chat\s*\(/)
  assert.doesNotMatch(source, /bot\.respawn\s*\(/)
})

test('action identifiers have one stable mapping and reject coercion', () => {
  assert.deepEqual(ACTIONS, ['SEARCH', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE', 'PLACE_TABLE', 'APPROACH_TABLE', 'CRAFT_PICKAXE'])
  for (const action of [null, true, '1', -1, 8, 1.5, NaN]) assert.throws(() => validateAction(action))
  assert.equal(validateAction(7), 7)
})

test('resource observations aggregate wood species without fabricating materials', () => {
  assert.deepEqual(inventoryCounts({ oak_log: 2, birch_log: 1, cherry_planks: 3, pale_oak_planks: 4, stick: 2, crafting_table: 1, wooden_pickaxe: 1, dirt: 32, crimson_stem: 3 }), { logs: 3, planks: 7, sticks: 2, tables: 1, pickaxes: 1 })
  assert.deepEqual(inventoryCounts({}), { logs: 0, planks: 0, sticks: 0, tables: 0, pickaxes: 0 })
  for (const amount of [-1, 0.5, NaN, '2']) assert.throws(() => inventoryCounts({ oak_log: amount }))
})

test('observations require real complete Survival health, food, and coordinates', () => {
  const observation = { position: { x: 0.5, y: 67, z: -2 }, health: 20, food: 20, gamemode: 'survival' }
  assert.equal(validatePhysicalState(observation), observation)
  assert.doesNotThrow(() => validatePhysicalState({ ...observation, health: 0 }))
  for (const invalid of [{ health: undefined }, { food: null }, { health: 21 }, { position: { x: NaN, y: 67, z: -2 } }, { gamemode: 'creative' }]) assert.throws(() => validatePhysicalState({ ...observation, ...invalid }))
})

test('navigation helper cannot mine, scaffold, sprint, parkour, or leave its starting area', () => {
  class FakeMovements {
    constructor () { this.blocksToAvoid = new Set(); this.exclusionAreasStep = []; this.exclusionAreasPlace = [] }
  }
  const movements = safeMovements({ registry: { blocksByName: { water: { id: 1 }, lava: { id: 2 } } } }, FakeMovements, { x: 10, z: -10 })
  for (const flag of ['canDig', 'allow1by1towers', 'allowParkour', 'allowSprinting', 'allowFreeMotion', 'infiniteLiquidDropdownDistance']) assert.equal(movements[flag], false)
  assert.deepEqual(movements.scafoldingBlocks, [])
  assert(movements.blocksToAvoid.has(1) && movements.blocksToAvoid.has(2))
  assert.equal(movements.exclusionAreasStep[0]({ position: { x: 10, z: -10 } }), 0)
  assert.equal(movements.exclusionAreasStep[0]({ position: { x: 75, z: -10 } }), 100)
  assert.equal(movements.exclusionAreasPlace[0](), 100)
})

test('mixed plank recipes consume available species and reject missing sticks', () => {
  const plank = { id: 1, count: 1, metadata: null }
  const stick = { id: 3, count: 1, metadata: null }
  const empty = { id: -1, count: 1, metadata: null }
  const template = { inShape: [[plank, plank, plank], [empty, stick, empty], [empty, stick, empty]], requiresTable: true }
  let items = [{ type: 1, count: 1 }, { type: 2, count: 2 }, { type: 3, count: 2 }]
  const bot = { recipesFor: () => [], recipesAll: () => [template], registry: { items: { 1: { name: 'oak_planks' }, 2: { name: 'birch_planks' }, 3: { name: 'stick' } } }, inventory: { items: () => items } }
  const recipe = craftingRecipe(bot, { id: 7, name: 'wooden_pickaxe' }, {})
  assert.deepEqual(recipe.inShape[0].map(ingredient => ingredient.id), [1, 2, 2])
  assert.deepEqual(template.inShape[0].map(ingredient => ingredient.id), [1, 1, 1])
  items = [{ type: 1, count: 1 }, { type: 2, count: 2 }, { type: 3, count: 1 }]
  assert.equal(craftingRecipe(bot, { id: 7, name: 'wooden_pickaxe' }, {}), null)
})

test('blocked log paths rotate candidates and force exploration before retrying', () => {
  const memory = new SearchMemory()
  const first = { position: { x: 1, y: 64, z: 1 } }
  const second = { position: { x: 2, y: 64, z: 2 } }
  const candidates = [first, second]
  assert.equal(memory.next(candidates), first)
  memory.failed(first)
  assert.equal(memory.next(candidates), second)
  memory.failed(second)
  assert.equal(memory.next(candidates), null)
  for (let i = 0; i < 5; i++) assert.equal(memory.next(candidates), null)
  assert.equal(memory.next(candidates), first)
})

test('table placement can replace ordinary plants but not other solid blocks or liquids', () => {
  for (const name of ['air', 'short_grass', 'tall_grass', 'fern', 'snow']) assert.equal(replaceableForTable(name), true)
  for (const name of ['oak_log', 'stone', 'water', 'lava', 'crafting_table', undefined]) assert.equal(replaceableForTable(name), false)
})

test('configured descent allows three feet-height blocks but rejects a four-block fall', () => {
  const { Movements } = require('mineflayer-pathfinder')
  class FakeMovements {
    constructor () { this.blocksToAvoid = new Set(); this.exclusionAreasStep = []; this.exclusionAreasPlace = [] }
  }
  const settings = safeMovements({ registry: { blocksByName: {} } }, FakeMovements, { x: 0, z: 0 })
  function landing (supportY) {
    const movement = {
      maxDropDown: settings.maxDropDown,
      bot: { game: { minY: -64 } },
      getBlock: (position, dx, dy, dz) => {
        const actual = { x: position.x + dx, y: position.y + dy, z: position.z + dz }
        return { position: actual, physical: actual.y === supportY, safe: actual.y !== supportY, liquid: false }
      }
    }
    return Movements.prototype.getLandingBlock.call(movement, { x: 0, y: 75, z: 0 }, { x: 1, z: 0 })
  }
  assert.equal(landing(71).position.y, 72)
  assert.equal(landing(70), null)
})
