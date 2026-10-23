'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const { parseArgs, validateAction, inventoryCounts, validatePhysicalState, craftingRecipe, safeMovements, SearchMemory, replaceableForTable, isLeaves, fallDamage, descentAcceptable, planCanopyDescent, nodeCollectsItem, planLeafEscape, explorationTarget, underHitbox, escapeNote, canSeeFace, eyesAt, hitboxOverlapsCell, placementCandidates, approachCells, interactiveSupport, stepExclusion, ACTIONS, FALL_HAZARDS, MAX_CANOPY_LAYERS, PICKUP_REACH, MAX_ESCAPE_LEAVES, EXPLORATION_RINGS, EXPLORATION_DIRECTIONS, MAX_RADIUS, TARGET_MARGIN, STEP_MARGIN, APPROACH_OFFSETS } = require('./resource-bridge')
const { Vec3 } = require('vec3')

// A small offline 1.21.4 world built from the same libraries Mineflayer uses,
// so visibility, placement and approach rules are tested on real block shapes.
const mineflayerEntry = require.resolve('mineflayer')
const fromMineflayer = name => require(require.resolve(name, { paths: [mineflayerEntry] }))
async function buildWorld (blocks) {
  const version = '1.21.4'
  const World = fromMineflayer('prismarine-world')(version)
  const Chunk = fromMineflayer('prismarine-chunk')(version)
  const Block = fromMineflayer('prismarine-block')(version)
  const registry = fromMineflayer('prismarine-registry')(version)
  const world = new World(null)
  const columns = new Set()
  for (const [x, , z] of blocks) {
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) columns.add(`${Math.floor(x / 16) + dx},${Math.floor(z / 16) + dz}`)
  }
  for (const key of columns) {
    const [cx, cz] = key.split(',').map(Number)
    await world.setColumn(cx, cz, new Chunk({ minY: -64, worldHeight: 384 }))
  }
  for (const [x, y, z, name, properties] of blocks) {
    // fromProperties({}) returns a block's first state, not its default.
    const state = properties ? Block.fromProperties(name, properties, 0).stateId : registry.blocksByName[name].defaultState
    world.sync.setBlockStateId(new Vec3(x, y, z), state)
  }
  return world.sync
}
function layer (x0, x1, z0, z1, y, name, skip = () => false) {
  const blocks = []
  for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) if (!skip(x, z)) blocks.push([x, y, z, name])
  return blocks
}
// Function-call arguments in the bridge source with line comments removed, so
// a comment can never satisfy a check about the code.
function callArguments (source, name) {
  const code = source.replace(/\/\/[^\n]*/g, '')
  const found = []
  let index = 0
  while ((index = code.indexOf(`${name}(`, index)) !== -1) {
    if (/[\w$]/.test(code[index - 1] || '')) { index += name.length; continue }
    let depth = 0
    let end = index + name.length
    for (; end < code.length; end++) {
      if (code[end] === '(') depth++
      else if (code[end] === ')' && --depth === 0) break
    }
    found.push(code.slice(index + name.length + 1, end))
    index = end
  }
  return found
}

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
  // Walkable cells keep their centre at least 1.5 blocks inside the radius.
  assert.equal(movements.exclusionAreasStep[0]({ position: { x: 10 + 61, z: -10 } }), 0) // centre 61.5 blocks out
  assert.equal(movements.exclusionAreasStep[0]({ position: { x: 10 + 63, z: -10 } }), 100) // centre 63.5
  // Radial, measured at the cell centre, threshold 62.5, with a half-block spawn.
  const spawn = { x: 0.5, z: 0.5 }
  assert.equal(stepExclusion(spawn, { x: 45, z: 44 }), 100) // diagonal, centre 62.94
  assert.equal(stepExclusion(spawn, { x: 44, z: 44 }), 0) // centre 62.23
  assert.equal(stepExclusion(spawn, { x: -62, z: 3 }), 0) // centre 62.07, min corner 62.55
  assert.equal(stepExclusion(spawn, { x: 62, z: 9 }), 100) // centre 62.65
  assert.equal(STEP_MARGIN, 1.5)
  // A bot pushed out to 63.6 may use its own column and step back inward, never outward.
  const pushed = { x: 64.1, z: 0.5 }
  assert.equal(stepExclusion(spawn, { x: 64, z: 0 }, pushed), 0) // its own column (centre 64.0)
  assert.equal(stepExclusion(spawn, { x: 63, z: 0 }, pushed), 0) // inward, centre 63.0
  assert.equal(stepExclusion(spawn, { x: 63, z: 1 }, pushed), 0) // inward, centre 63.01
  assert.equal(stepExclusion(spawn, { x: 65, z: 0 }, pushed), 100) // outward
  assert.equal(stepExclusion(spawn, { x: 63, z: 8 }, pushed), 100) // centre 63.5: past the 63.25 ceiling
  const live = safeMovements({ registry: { blocksByName: {} }, entity: { position: pushed } }, FakeMovements, spawn)
  assert.equal(live.exclusionAreasStep[0]({ position: { x: 63, z: 0 } }), 0)
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
    assert(Math.hypot(t.dx * t.radius, t.dz * t.radius) + TARGET_MARGIN <= MAX_RADIUS, `target ${i} leaves the search area`)
  }
  assert.equal(explorationTarget(8 * EXPLORATION_RINGS - 1).radius, 43)
  for (const bad of [-1, 1.5, '3', null]) assert.throws(() => explorationTarget(bad))
  assert(TARGET_MARGIN >= 2 && MAX_RADIUS === 64)
})

test('CHOP and SEARCH never target a block beneath the bot and every helper target keeps the boundary margin', () => {
  const bot = { x: 9.5, y: 87, z: 32.5 }
  assert.equal(underHitbox(bot, { x: 9, y: 86, z: 32 }), true) // its support block
  assert.equal(underHitbox(bot, { x: 9, y: 80, z: 32 }), true) // anything deeper in its column
  assert.equal(underHitbox(bot, { x: 10, y: 86, z: 32 }), false) // the trunk beside it
  assert.equal(underHitbox(bot, { x: 9, y: 87, z: 32 }), false) // not below the feet
  // Standing on an edge, the corner column also carries the bot.
  assert.equal(underHitbox({ x: 9.75, y: 87, z: 32.5 }, { x: 10, y: 86, z: 32 }), true)
  // Negative coordinates (the PC seed-10002 trunk) and a partial block holding the feet.
  assert.equal(underHitbox({ x: -4.5, y: 94, z: -2.5 }, { x: -5, y: 93, z: -3 }), true)
  assert.equal(underHitbox({ x: -4.5, y: 94, z: -2.5 }, { x: -4, y: 93, z: -3 }), false)
  assert.equal(underHitbox({ x: 9.5, y: 63.9375, z: 32.5 }, { x: 9, y: 63, z: 32 }), true)
  assert.equal(underHitbox({ x: 9.5, y: 63.9375, z: 32.5 }, { x: 9, y: 64, z: 32 }), false)

  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  const code = source.replace(/\/\/[^\n]*/g, '')
  // reachableLogs, CHOP's re-check, and trunkTargets all apply the rule.
  assert.equal((code.match(/underHitbox\(bot\.entity\.position, block\.position\)/g) || []).length, 3)
  // The library's single-ray visibility is not used anywhere; one wrapper calls the face check.
  assert.equal((code.match(/bot\.canSeeBlock\s*\(/g) || []).length, 0)
  assert.equal(callArguments(source, 'canSeeFace').filter(args => args.startsWith('bot.world')).length, 1)
  // Every inBounds call keeps the margin except the hard check on each observation.
  const bounds = callArguments(source, 'inBounds')
  assert(bounds.length >= 7, `only ${bounds.length} inBounds calls found`)
  assert.deepEqual(bounds.filter(args => !/, TARGET_MARGIN$/.test(args)).sort(), ['drop.position, STEP_MARGIN', 'result.position'])
  assert.equal(TARGET_MARGIN, 3)
  // The pure placement and approach helpers use the same margin.
  assert.deepEqual(callArguments(source, 'withinRadius').filter(args => !/, TARGET_MARGIN$/.test(args)), ['spawn, position, margin'])
})

test('a table diagonal to a bot standing in a chopped trunk is seen past leaves at head height', async () => {
  // Modelled on pc2-sarsa-frozen-val/episode-0006 (a table diagonal to a bot
  // in a chopped trunk column). The retained transitions do not record every
  // surrounding block, so this is a representative hard case, not a replica:
  // grass at y 66, the trunk column (-50, 85) chopped from 67 to 69, its log at
  // 70, oak leaves around it at 68 and 69, and the table at (-49, 67, 86).
  const trunk = (x, z) => x === -50 && z === 85
  const world = await buildWorld([
    ...layer(-53, -46, 82, 89, 66, 'grass_block'),
    ...layer(-52, -48, 83, 87, 68, 'oak_leaves', trunk),
    ...layer(-52, -48, 83, 87, 69, 'oak_leaves', trunk),
    [-50, 70, 85, 'oak_log'],
    [-49, 67, 86, 'crafting_table']
  ])
  const table = world.getBlock(new Vec3(-49, 67, 86))
  for (const bot of [new Vec3(-49.62, 67, 85.68), new Vec3(-49.5, 67, 85.5), new Vec3(-49.35, 67, 85.35)]) {
    assert.equal(canSeeFace(world, eyesAt(bot), table), true, `table hidden from ${bot}`)
  }
  // Its top face is covered by the leaf block above and is not counted as seen.
  assert.equal(canSeeFace(world, eyesAt(new Vec3(-49.5, 67, 85.5)), table, { faces: ['top'] }), false)
})

test('visibility respects walls, real collision shapes, and ignores shapes behind the eyes', async () => {
  const floor = layer(-2, 6, -2, 2, 63, 'grass_block')
  const bot = new Vec3(0.5, 64, 0.5)
  const open = await buildWorld([...floor, [3, 64, 0, 'crafting_table']])
  assert.equal(canSeeFace(open, eyesAt(bot), open.getBlock(new Vec3(3, 64, 0))), true)
  const walled = await buildWorld([...floor, [2, 64, 0, 'stone'], [2, 65, 0, 'stone'], [3, 64, 0, 'crafting_table']])
  assert.equal(canSeeFace(walled, eyesAt(bot), walled.getBlock(new Vec3(3, 64, 0))), false)
  // A bottom slab behind a one-block step: the ray enters the slab's empty
  // upper half, but its shape and its top face are hidden by the step.
  const stepped = await buildWorld([...layer(-1, 4, -1, 1, 64, 'grass_block'), [1, 65, 0, 'dirt'], [2, 65, 0, 'oak_slab']])
  const slab = stepped.getBlock(new Vec3(2, 65, 0))
  assert.deepEqual(slab.shapes, [[0, 0, 0, 1, 0.5, 1]])
  assert.equal(canSeeFace(stepped, eyesAt(new Vec3(0.5, 65, 0.5)), slab), false)
  assert.equal(canSeeFace(stepped, eyesAt(new Vec3(0.5, 65, 0.5)), slab, { faces: ['top'] }), false)
  const flat = await buildWorld([...layer(-1, 4, -1, 1, 64, 'grass_block'), [2, 65, 0, 'oak_slab']])
  assert.equal(canSeeFace(flat, eyesAt(new Vec3(0.5, 65, 0.5)), flat.getBlock(new Vec3(2, 65, 0)), { faces: ['top'] }), true)
  // A closed top-half trapdoor in the head cell lies above the eyes; the
  // library reports it at a negative distance, which must not hide the table.
  const trapdoor = await buildWorld([...floor, [0, 65, 0, 'oak_trapdoor', { facing: 'north', half: 'top', open: false, powered: false, waterlogged: false }], [1, 64, 0, 'crafting_table']])
  assert.deepEqual(trapdoor.getBlock(new Vec3(0, 65, 0)).shapes, [[0, 0.8125, 0, 1, 1, 1]])
  assert.equal(canSeeFace(trapdoor, eyesAt(bot), trapdoor.getBlock(new Vec3(1, 64, 0))), true)
  assert.equal(canSeeFace(open, eyesAt(bot), open.getBlock(new Vec3(5, 70, 0))), false) // air has no shape
})

test('placement never chooses a cell the hitbox overlaps and prefers an orthogonal neighbour', async () => {
  const world = await buildWorld([...layer(4, 15, 27, 38, 63, 'grass_block'), [10, 64, 32, 'stone'], [9, 64, 33, 'stone']])
  const spawn = { x: 9.5, z: 32.5 }
  const overlaps = (bot, cell) => Math.abs(cell.x + 0.5 - bot.x) < 0.8 && Math.abs(cell.z + 0.5 - bot.z) < 0.8
  // Off-centre bots, including the review's (9.75, 32.75) case where the old
  // 0.9-block radius test picked the overlapped diagonal cell first.
  for (const bot of [new Vec3(9.75, 64, 32.75), new Vec3(9.71, 64, 32.95), new Vec3(9.25, 64, 32.25), new Vec3(9.5, 64, 32.5)]) {
    const candidates = placementCandidates(world, bot, spawn, { radius: 3, withinReach: true })
    assert(candidates.length > 0)
    for (const { position } of candidates) {
      assert.equal(overlaps(bot, position), false, `${position} overlaps the bot at ${bot}`)
      assert.equal(hitboxOverlapsCell(bot, position), false)
    }
    // Orthogonal to the bot's cell: at feet level where free, otherwise on top
    // of the stone beside it (at (9.25, 32.25) both open feet-level orthogonal
    // cells overlap the hitbox).
    const first = candidates[0].position
    assert.equal(Math.abs(first.x - Math.floor(bot.x)) + Math.abs(first.z - Math.floor(bot.z)), 1, `first spot ${first} is not orthogonal for ${bot}`)
  }
  // With +x and +z blocked by stone, a centred bot still gets -x or -z, never the diagonal.
  const first = placementCandidates(world, new Vec3(9.5, 64, 32.5), spawn, { radius: 3, withinReach: true })[0].position
  assert(['8,64,32', '9,64,31'].includes(`${first.x},${first.y},${first.z}`), `${first}`)
  // Nothing is offered outside the margin.
  for (const { position } of placementCandidates(world, new Vec3(9.5, 64, 32.5), { x: 9.5 - 61, z: 32.5 }, { radius: 3, withinReach: false })) {
    assert(Math.hypot(position.x - (9.5 - 61), position.z - 32.5) <= MAX_RADIUS - TARGET_MARGIN)
  }
})

test('table approach cells follow the pathfinder: level first, carpets passable, fences not floors', async () => {
  const world = await buildWorld([...layer(2, 8, 2, 8, 63, 'grass_block'), [5, 64, 5, 'crafting_table'], [6, 65, 5, 'oak_leaves'],
    [5, 63, 6, 'water'], [4, 64, 5, 'white_carpet'], [5, 63, 4, 'oak_fence']])
  const cells = approachCells(world, new Vec3(5, 64, 5), { x: 5.5, z: 5.5 }).map(cell => `${cell.x},${cell.y},${cell.z}`)
  // +x has leaves at head height; +z stands over water; -z stands on a fence;
  // -x holds a carpet, which the bot walks through; the diagonals qualify.
  assert.deepEqual(cells, ['4,64,5', '6,64,6', '6,64,4', '4,64,6', '4,64,4'])
  assert.deepEqual(APPROACH_OFFSETS.slice(0, 4), [[1, 0], [-1, 0], [0, 1], [0, -1]])
  // A table on a one-block bump is approached from the ground beside it.
  const bump = await buildWorld([...layer(6, 13, 29, 36, 63, 'grass_block'), [10, 64, 32, 'dirt'], [10, 65, 32, 'crafting_table']])
  assert.deepEqual(approachCells(bump, new Vec3(10, 65, 32), { x: 9.5, z: 32.5 }).map(cell => `${cell.x},${cell.y},${cell.z}`),
    ['11,64,32', '9,64,32', '10,64,33', '10,64,31', '11,64,33', '11,64,31', '9,64,33', '9,64,31'])
})

test('placement skips blocks a click would use, creatures, the escape exit, and prefers level cells', async () => {
  const spawn = { x: 9.5, z: 32.5 }
  const centred = new Vec3(9.5, 64, 32.5)
  const pillars = [[10, 64, 32], [10, 65, 32], [9, 64, 33], [9, 65, 33], [9, 64, 31], [9, 65, 31]].map(([x, y, z]) => [x, y, z, 'oak_planks'])
  for (const name of ['chest', 'barrel', 'furnace', 'oak_trapdoor', 'red_bed', 'lectern', 'cake', 'white_shulker_box', 'oak_door']) {
    assert.equal(interactiveSupport(name), true, name)
    const world = await buildWorld([...layer(5, 14, 28, 37, 63, 'grass_block'), ...pillars, [8, 64, 32, name]])
    const candidates = placementCandidates(world, centred, spawn, { radius: 3, withinReach: true })
    assert(candidates.every(candidate => candidate.support.name !== name), `${name} offered as a support`)
  }
  for (const name of ['grass_block', 'stone', 'oak_planks', 'oak_log', 'oak_leaves', 'dirt']) assert.equal(interactiveSupport(name), false, name)
  const open = await buildWorld(layer(5, 14, 28, 37, 63, 'grass_block'))
  // A cow standing in the west neighbour cell keeps the table out of it.
  const cow = { x0: 8.05, x1: 8.95, y0: 64, y1: 65.4, z0: 32.05, z1: 32.95 }
  const withCow = placementCandidates(open, centred, spawn, { radius: 3, withinReach: true, obstacles: [cow] })
  assert(withCow.length > 0 && withCow.every(c => !(c.position.x === 8 && c.position.y === 64 && c.position.z === 32)))
  // The column an escape just opened is never filled.
  const kept = placementCandidates(open, centred, spawn, { radius: 3, withinReach: true, excludedColumns: [{ x: 10, z: 32 }] })
  assert(kept.length > 0 && kept.every(c => !(c.position.x === 10 && c.position.z === 32)))
  // A level orthogonal cell beats a raised one even when the raised one is nearer.
  const raised = await buildWorld([...layer(5, 14, 28, 37, 63, 'grass_block'), [10, 64, 32, 'stone'], [9, 64, 33, 'stone'], [9, 64, 31, 'stone']])
  const first = placementCandidates(raised, new Vec3(9.7, 64, 32.5), spawn, { radius: 3, withinReach: true })[0].position
  assert.equal(`${first.x},${first.y},${first.z}`, '8,64,32')
  // Every in-reach candidate's support top is within reach; farther ones exist only without that filter.
  const eyes = new Vec3(9.5, 65.62, 32.5)
  const near = placementCandidates(open, centred, spawn, { radius: 3, withinReach: true })
  assert(near.every(c => c.support.position.offset(0.5, 1, 0.5).distanceTo(eyes) <= 4.3))
  assert(placementCandidates(open, centred, spawn, { radius: 3, withinReach: false }).some(c => c.support.position.offset(0.5, 1, 0.5).distanceTo(eyes) > 4.3))
})

test('a foliage escape is named in the event of whichever action ran it', () => {
  assert.equal(escapeNote({ blocks_dug: ['oak_leaves'], stopped: null }), 'cleared_1_leaf_wall_block')
  assert.equal(escapeNote({ blocks_dug: ['oak_leaves', 'oak_leaves'], stopped: null }), 'cleared_2_leaf_wall_blocks')
  assert.equal(escapeNote({ blocks_dug: [], stopped: 'no_leaf_exit' }), 'leaf_escape_refused:no_leaf_exit')
  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  const place = source.slice(source.indexOf('async function placeTable'), source.indexOf('async function step'))
  assert.match(place, /const escape = await escapeFoliage\(signal\)\s+note = escapeNote\(escape\)/)
  assert.match(place, /if \(escape\.exit && escape\.blocks_dug\.length\) excludedColumns = \[escape\.exit\]/)
  assert.match(place, /return label\('table_placed'\)/)
  assert.doesNotMatch(place, /return 'table_placed'|return 'no_placement_surface'/)
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
