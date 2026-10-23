'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const { parseArgs, validateAction, inventoryCounts, validatePhysicalState, craftingRecipe, safeMovements, SearchMemory, replaceableForTable, isLeaves, fallDamage, descentAcceptable, planCanopyDescent, nodeCollectsItem, planLeafEscape, explorationTarget, ACTIONS, FALL_HAZARDS, MAX_CANOPY_LAYERS, PICKUP_REACH, MAX_ESCAPE_LEAVES, EXPLORATION_RINGS, EXPLORATION_DIRECTIONS } = require('./resource-bridge')

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
  // A trunk stays blocked whichever of its logs happens to be visible next time.
  const trunk = new SearchMemory()
  trunk.next([{ position: { x: 168, y: 70, z: 160 } }])
  trunk.failed({ position: { x: 168, y: 70, z: 160 } })
  assert.equal(trunk.next([{ position: { x: 168, y: 71, z: 160 } }]), null)
})

test('a pickup goal ends only where the vanilla item box can reach the bot', () => {
  assert(PICKUP_REACH < 1.425 - 0.2)
  const item = { x: 10.05, y: 64, z: 20.5 }
  assert.equal(nodeCollectsItem({ x: 10, y: 64, z: 20 }, item), true)
  assert.equal(nodeCollectsItem({ x: 10, y: 64, z: 21 }, item), true)
  assert.equal(nodeCollectsItem({ x: 9, y: 64, z: 19 }, item), true)
  // Orthogonally adjacent on the far side: 1.45 blocks away, outside the pickup box.
  assert.equal(nodeCollectsItem({ x: 11, y: 64, z: 20 }, item), false)
  // One block above the feet is still collected; one block below is not.
  assert.equal(nodeCollectsItem({ x: 10, y: 63, z: 20 }, item), true)
  assert.equal(nodeCollectsItem({ x: 10, y: 65, z: 20 }, item), false)
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

test('canopy descent digs only the leaf block under the feet after verifying the landing', () => {
  const leaves = { name: 'oak_leaves', boundingBox: 'block' }
  const air = { name: 'air', boundingBox: 'empty' }
  const grass = { name: 'grass_block', boundingBox: 'block' }
  // A typical oak canopy: another leaf layer one block down, then open air to the ground.
  assert.deepEqual(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, leaves, air, air, air, grass] }),
    { ok: true, dig_block: 'oak_leaves', landing_y: 75, landing_block: 'oak_leaves', fall: 1, expected_damage: 0 })
  const second = planCanopyDescent({ feetY: 75, health: 20, column: [leaves, air, air, air, grass] })
  assert.deepEqual(second, { ok: true, dig_block: 'oak_leaves', landing_y: 71, landing_block: 'grass_block', fall: 4, expected_damage: 1 })
  // The same step is refused when its damage would drop health into the low bin.
  assert.deepEqual(planCanopyDescent({ feetY: 75, health: 11, column: [leaves, air, air, air, grass] }),
    { ok: false, reason: 'fall_too_far', fall: 4, expected_damage: 1, landing_block: 'grass_block' })
  // Two points of damage is the limit even at full health.
  const six = planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, air, air, air, air, grass] })
  assert.equal(six.ok, false)
  assert.equal(six.reason, 'fall_too_far')
  assert.equal(six.fall, 6)
  assert.equal(six.expected_damage, 3)
  // Landings are measured to the block's collision surface: a planned five-block
  // fall onto a carpet or lily pad is really 5.9 blocks and three points, so it is refused.
  const carpet = { name: 'pale_moss_carpet', boundingBox: 'block', top: 0.0625 }
  const mud = { name: 'mud', boundingBox: 'block', top: 0.875 }
  const lilyPad = { name: 'lily_pad', boundingBox: 'block', top: 0.09375 }
  const ontoCarpet = planCanopyDescent({ feetY: 76, health: 13, column: [leaves, air, air, air, air, carpet, grass] })
  assert.deepEqual([ontoCarpet.ok, ontoCarpet.reason, ontoCarpet.fall, ontoCarpet.expected_damage], [false, 'fall_too_far', 5.9375, 3])
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, air, air, air, carpet, grass] }).ok, false)
  assert.equal(planCanopyDescent({ feetY: 76, health: 13, column: [leaves, air, air, air, air, lilyPad, { name: 'water', boundingBox: 'empty' }] }).reason, 'fall_too_far')
  const ontoMud = planCanopyDescent({ feetY: 75, health: 20, column: [leaves, air, air, air, mud] })
  assert.deepEqual([ontoMud.ok, ontoMud.fall, ontoMud.expected_damage, ontoMud.landing_y], [true, 4.125, 2, 70.875])
  assert.equal(planCanopyDescent({ feetY: 75, health: 12, column: [leaves, air, air, air, mud] }).reason, 'fall_too_far')
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, air, { name: 'big_dripleaf', boundingBox: 'block', top: 0.9375 }] }).reason, 'hazard_below')
  // Anything other than a leaf block under the feet is never dug by this helper.
  assert.equal(planCanopyDescent({ feetY: 70, health: 20, column: [grass, { name: 'dirt', boundingBox: 'block' }] }).reason, 'not_standing_on_leaves')
  assert.equal(planCanopyDescent({ feetY: 70, health: 20, column: [{ name: 'oak_log', boundingBox: 'block' }, grass] }).reason, 'not_standing_on_leaves')
  assert.equal(planCanopyDescent({ feetY: 70, health: 20, column: [air, grass] }).reason, 'not_standing_on_leaves')
  // Hazards in the gap or as the landing, unloaded chunks, and bottomless columns all refuse.
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, { name: 'water', boundingBox: 'empty' }, grass] }).reason, 'hazard_below')
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, { name: 'magma_block', boundingBox: 'block' }] }).reason, 'hazard_below')
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, { name: 'powder_snow', boundingBox: 'empty' }, grass] }).reason, 'hazard_below')
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, null, grass] }).reason, 'unloaded_column')
  assert.equal(planCanopyDescent({ feetY: 76, health: 20, column: [leaves, air, air] }).reason, 'no_landing_within_scan')
  for (const bad of [{ feetY: 76.5, health: 20, column: [leaves, grass] }, { feetY: 76, health: 21, column: [leaves, grass] }, { feetY: 76, health: NaN, column: [leaves, grass] }, { feetY: 76, health: 20, column: [leaves] }, { feetY: 76, health: 20, column: 'none' }]) {
    assert.throws(() => planCanopyDescent(bad))
  }
  for (const name of ['water', 'lava', 'powder_snow', 'sweet_berry_bush', 'cactus', 'magma_block']) assert(FALL_HAZARDS.includes(name))
  assert(Number.isInteger(MAX_CANOPY_LAYERS) && MAX_CANOPY_LAYERS >= 1 && MAX_CANOPY_LAYERS <= 8)
})

test('fall damage follows vanilla ceil(distance - 3) and the descent health rule', () => {
  assert.deepEqual([0, 3, 3.5, 4, 5, 6].map(fallDamage), [0, 0, 1, 1, 2, 3])
  assert.throws(() => fallDamage(-1))
  assert.throws(() => fallDamage(NaN))
  assert.equal(descentAcceptable(3, 1), true)
  assert.equal(descentAcceptable(5, 20), true)
  assert.equal(descentAcceptable(5, 12), false)
  assert.equal(descentAcceptable(5, 13), true)
  assert.equal(descentAcceptable(4, 11), false)
  assert.equal(descentAcceptable(6, 20), false)
  for (const name of ['oak_leaves', 'azalea_leaves', 'flowering_azalea_leaves', 'pale_oak_leaves']) assert.equal(isLeaves(name), true)
  for (const name of ['oak_log', 'leaves', 'grass_block', undefined, null, 4]) assert.equal(isLeaves(name), false)
})

test('a foliage escape clears only leaf blocks towards a floored exit and prefers the cheapest', () => {
  const air = { name: 'air', boundingBox: 'empty' }
  const leaves = { name: 'oak_leaves', boundingBox: 'block' }
  const grass = { name: 'grass_block', boundingBox: 'block' }
  const log = { name: 'oak_log', boundingBox: 'block' }
  const water = { name: 'water', boundingBox: 'empty' }
  // The PC seed-10006 shaft as the saved world shows it: leaves at head height
  // all round, leaves at feet height on two sides, grass at feet height on the
  // other two (a grass wall is never dug, so a two-leaf side is chosen).
  const shaft = {
    '+x': { feet: grass, head: leaves, below: grass, lower: grass },
    '-x': { feet: leaves, head: leaves, below: grass, lower: grass },
    '+z': { feet: grass, head: leaves, below: grass, lower: grass },
    '-z': { feet: leaves, head: leaves, below: grass, lower: grass }
  }
  assert.deepEqual(planLeafEscape(shaft), { ok: true, direction: '-x', dig: ['oak_leaves', 'oak_leaves'], floor: 'grass_block' })
  const boxed = {
    '+x': { feet: leaves, head: leaves, below: grass, lower: grass },
    '-x': { feet: leaves, head: air, below: air, lower: grass }, // one step down, one leaf
    '+z': { feet: log, head: leaves, below: grass, lower: grass }, // trunk: never dug
    '-z': { feet: leaves, head: leaves, below: water, lower: grass } // hazard floor
  }
  assert.deepEqual(planLeafEscape(boxed), { ok: true, direction: '-x', dig: ['oak_leaves'], floor: 'grass_block' })
  assert.deepEqual(planLeafEscape({ '+x': boxed['+x'] }), { ok: true, direction: '+x', dig: ['oak_leaves', 'oak_leaves'], floor: 'grass_block' })
  assert.equal(planLeafEscape({ '+z': boxed['+z'], '-z': boxed['-z'] }).reason, 'no_leaf_exit')
  // Whether the bot is walled in at all is decided by the pathfinder's own move rules.
  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  assert.match(source, /bot\.pathfinder\.movements\.getNeighbors\(new Move\(/)
  assert.match(source, /if \(recovery\) recovery\.escape = record; else recovery = record/)
  // Bottomless, unloaded, out-of-bounds, or already open neighbours are no exits.
  assert.equal(planLeafEscape({ '+x': { feet: leaves, head: leaves, below: air, lower: air } }).reason, 'no_leaf_exit')
  assert.equal(planLeafEscape({ '+x': { feet: leaves, head: null, below: grass, lower: grass }, '-x': null }).reason, 'no_leaf_exit')
  assert.equal(planLeafEscape({ '+x': { feet: air, head: air, below: grass, lower: grass } }).reason, 'no_leaf_exit')
  assert.equal(planLeafEscape({}).reason, 'no_leaf_exit')
  assert(Number.isInteger(MAX_ESCAPE_LEAVES) && MAX_ESCAPE_LEAVES >= 1 && MAX_ESCAPE_LEAVES <= 2)
})

test('exploration rings cover every heading before growing and stay inside the search area', () => {
  const first = Array.from({ length: 8 }, (_, i) => explorationTarget(i))
  assert.deepEqual(first.map(t => [t.dx, t.dz]), EXPLORATION_DIRECTIONS)
  assert(first.every(t => t.radius === 8))
  assert.equal(explorationTarget(8).radius, 16)
  assert.equal(explorationTarget(8 * (EXPLORATION_RINGS - 1)).radius, 8 * EXPLORATION_RINGS) // cardinal of the last ring
  assert.equal(explorationTarget(8 * EXPLORATION_RINGS).radius, 8)
  // Every target, plus the goal's two-block tolerance, stays inside the 64-block radius,
  // including the diagonals of the outer ring (48 * sqrt 2 would be 67.9).
  for (let i = 0; i < 8 * EXPLORATION_RINGS; i++) {
    const t = explorationTarget(i)
    assert(Math.hypot(t.dx * t.radius, t.dz * t.radius) + 2 <= 64, `target ${i} leaves the search area`)
  }
  assert.equal(explorationTarget(8 * EXPLORATION_RINGS - 1).radius, 43)
  for (const bad of [-1, 1.5, '3', null]) assert.throws(() => explorationTarget(bad))
})

test('CHOP digs a reachable log from where the bot stands and only walks for the drop', () => {
  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  const chop = source.slice(source.indexOf('async function chop'), source.indexOf('async function craft'))
  assert.equal((chop.match(/await digBlock\s*\(/g) || []).length, 1)
  // Up to three already-reachable candidates, and no approach move before the
  // dig: walking can hide a trunk's bottom log behind the log above it, which
  // stranded a frozen policy in sixty identical CHOP failures on the PC.
  assert.match(chop, /reachableLogs\(\)\.slice\(0, 3\)/)
  assert(chop.indexOf('navigate(') > chop.indexOf('await digBlock('))
  assert.match(chop, /actionDetail = detail/)
})

test('the bridge digs only through its chop and canopy helpers and still never issues commands', () => {
  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  // One bounded, health-guarded dig helper, called from CHOP, the canopy descent, and the foliage escape only.
  assert.equal((source.match(/bot\.dig\s*\(/g) || []).length, 1)
  assert.equal((source.match(/await digBlock\s*\(/g) || []).length, 3)
  assert.doesNotMatch(source, /bot\.chat\s*\(/)
  assert.doesNotMatch(source, /bot\.creative/)
  assert.doesNotMatch(source, /setControlState\s*\(\s*'jump'/)
  // The pathfinder may still never dig, place, or scaffold on its own.
  assert.match(source, /movements\.canDig = false/)
  assert.match(source, /movements\.scafoldingBlocks = \[\]/)
})
