'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawn, spawnSync } = require('node:child_process')
const { parseArgs, validateAction, inventoryCounts, validatePhysicalState, craftingRecipe, safeMovements, SearchMemory, replaceableForTable, isLeaves, fallDamage, descentAcceptable, planCanopyDescent, nodeCollectsItem, planLeafEscape, explorationTarget, underHitbox, escapeNote, canSeeFace, eyesAt, hitboxOverlapsCell, placementCandidates, approachCells, interactiveSupport, stepExclusion, ACTIONS, FALL_HAZARDS, MAX_CANOPY_LAYERS, PICKUP_REACH, MAX_ESCAPE_LEAVES, EXPLORATION_RINGS, EXPLORATION_DIRECTIONS, MAX_RADIUS, TARGET_MARGIN, STEP_MARGIN, APPROACH_OFFSETS,
  WOOD_ACTIONS, GOALS, STONE_LIT_ONLY, SURVEY_STAIR_STEPS, actionsFor, runnableAction, goalReached, pickaxeUsesLeft, stoneScan, depthToStone, surveyStaircases, stoneSurveyRecord } = require('./resource-bridge')
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
  assert.deepEqual(parseArgs([]), { host: '127.0.0.1', port: 25567, username: 'RLResourceBot', version: '1.21.4', goal: 'wooden_pickaxe', stoneSurvey: false })
  assert.equal(parseArgs(['--port', '25568']).port, 25568)
  for (const args of [['--host', 'example.com'], ['--port', '0'], ['--port', '2.5'], ['--username', 'bad name'], ['--version', '1.21.5'], ['--dev-arena']]) assert.throws(() => parseArgs(args))
  const source = fs.readFileSync(require.resolve('./resource-bridge'), 'utf8')
  assert.doesNotMatch(source, /bot\.chat\s*\(/)
  assert.doesNotMatch(source, /bot\.respawn\s*\(/)
})

test('action identifiers have one stable mapping and reject coercion', () => {
  assert.deepEqual(ACTIONS, ['SEARCH', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE', 'PLACE_TABLE', 'APPROACH_TABLE', 'CRAFT_PICKAXE',
    'SEARCH_STONE', 'DIG_DOWN', 'MINE_STONE', 'RETURN_TO_TABLE', 'CRAFT_STONE_PICKAXE'])
  // The wood stage keeps indices 0 to 7, so the frozen wood checkpoint keeps its meaning.
  assert.deepEqual(WOOD_ACTIONS, ['SEARCH', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE', 'PLACE_TABLE', 'APPROACH_TABLE', 'CRAFT_PICKAXE'])
  assert.deepEqual(actionsFor('wooden_pickaxe'), WOOD_ACTIONS)
  assert.deepEqual(actionsFor('stone_pickaxe'), ACTIONS)
  for (const action of [null, true, '1', -1, 8, 1.5, NaN]) assert.throws(() => validateAction(action))
  assert.equal(validateAction(7), 7)
  assert.throws(() => actionsFor('iron_pickaxe'), /Unknown goal/)
})

test('goal gating: the wood goal refuses actions 8 to 12 as it always refused out-of-range actions', () => {
  // The default goal's message is the one the bridge has always given, for every index past 7.
  for (const goal of [undefined, 'wooden_pickaxe']) {
    for (const action of [8, 9, 10, 11, 12, 13, -1, 1.5, '8']) {
      assert.throws(() => validateAction(action, goal), { message: 'action must be an integer from 0 through 7' })
      assert.throws(() => runnableAction(action, 'wooden_pickaxe'), { message: 'action must be an integer from 0 through 7' })
    }
  }
  for (let action = 0; action < 8; action++) {
    assert.equal(runnableAction(action, 'wooden_pickaxe'), action)
    assert.equal(runnableAction(action, 'stone_pickaxe'), action)
  }
  // The stone goal accepts 0 to 12; its five helpers are step S3, so until then each fails loudly by name.
  for (let action = 8; action <= 12; action++) {
    assert.equal(validateAction(action, 'stone_pickaxe'), action)
    assert.throws(() => runnableAction(action, 'stone_pickaxe'), { message: `${ACTIONS[action]} (action ${action}) is not implemented until step S3; this episode is invalid` })
  }
  for (const action of [13, -1, null, '9']) assert.throws(() => runnableAction(action, 'stone_pickaxe'), { message: 'action must be an integer from 0 through 12' })
  assert.throws(() => runnableAction(1, 'wood'), /Unknown goal/)
})

test('success follows the goal: any wooden pickaxe for wood, only a held stone pickaxe for stone', () => {
  const counts = pickaxes => ({ logs: 0, planks: 0, sticks: 0, tables: 0, pickaxes, stone_materials: 3, stone_pickaxes: 0 })
  assert.equal(goalReached('wooden_pickaxe', counts(1), null), true) // held or not, as at 87673b8
  assert.equal(goalReached('wooden_pickaxe', counts(1), 'wooden_pickaxe'), true)
  assert.equal(goalReached('wooden_pickaxe', counts(0), 'stone_pickaxe'), false)
  assert.equal(goalReached('stone_pickaxe', counts(1), 'wooden_pickaxe'), false)
  assert.equal(goalReached('stone_pickaxe', { ...counts(1), stone_pickaxes: 1 }, 'wooden_pickaxe'), false) // crafted but not held
  assert.equal(goalReached('stone_pickaxe', { ...counts(1), stone_pickaxes: 1 }, 'stone_pickaxe'), true)
  assert.throws(() => goalReached('diamond', counts(1), null), /Unknown goal/)
  assert.deepEqual(GOALS, ['wooden_pickaxe', 'stone_pickaxe'])
})

test('the goal and the survey option are strict command-line values', () => {
  assert.equal(parseArgs(['--goal', 'stone_pickaxe']).goal, 'stone_pickaxe')
  assert.equal(parseArgs(['--goal', 'wooden_pickaxe', '--stone-survey', 'on']).stoneSurvey, true)
  assert.equal(parseArgs(['--stone-survey', 'off', '--port', '25570']).stoneSurvey, false)
  assert.deepEqual(parseArgs(['--goal', 'wooden_pickaxe', '--stone-survey', 'off']), parseArgs([]))
  for (const args of [['--goal', 'stone'], ['--goal', 'Stone_Pickaxe'], ['--goal', ''], ['--goal'], ['--stone-survey', 'true'], ['--stone-survey', 'ON'],
    ['--stone-survey', '1'], ['--stone-survey'], ['--survey', 'on'], ['constructor', 'x'], ['__proto__', 'x'], ['toString', 'x']]) {
    assert.throws(() => parseArgs(args), undefined, args.join(' '))
  }
})

test('resource observations aggregate wood species without fabricating materials', () => {
  assert.deepEqual(inventoryCounts({ oak_log: 2, birch_log: 1, cherry_planks: 3, pale_oak_planks: 4, stick: 2, crafting_table: 1, wooden_pickaxe: 1, dirt: 32, crimson_stem: 3 }),
    { logs: 3, planks: 7, sticks: 2, tables: 1, pickaxes: 1, stone_materials: 0, stone_pickaxes: 0 })
  assert.deepEqual(inventoryCounts({}), { logs: 0, planks: 0, sticks: 0, tables: 0, pickaxes: 0, stone_materials: 0, stone_pickaxes: 0 })
  for (const amount of [-1, 0.5, NaN, '2']) assert.throws(() => inventoryCounts({ oak_log: amount }))
  // The stone counts follow the five wood counts; the wood counts never change because of stone.
  const stone = inventoryCounts({ cobblestone: 2, cobbled_deepslate: 1, blackstone: 4, stone: 5, granite: 3, mossy_cobblestone: 1, stone_pickaxe: 1, wooden_pickaxe: 1, stick: 2 })
  assert.deepEqual(Object.keys(stone), ['logs', 'planks', 'sticks', 'tables', 'pickaxes', 'stone_materials', 'stone_pickaxes'])
  assert.deepEqual(stone, { logs: 0, planks: 0, sticks: 2, tables: 0, pickaxes: 1, stone_materials: 7, stone_pickaxes: 1 })
  assert.throws(() => inventoryCounts({ cobblestone: -1 }), /Invalid inventory quantity/)
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
  // Step S1 adds no dig: the stone observation and the survey only read the world.
  assert.equal((source.match(/bot\.dig\s*\(/g) || []).length, 1)
  assert.equal((source.match(/await digBlock\s*\(/g) || []).length, 3)
  assert.doesNotMatch(source, /bot\.chat\s*\(/)
  assert.doesNotMatch(source, /bot\.creative/)
  assert.doesNotMatch(source, /setControlState\s*\(\s*'jump'/)
  // The pathfinder may still never dig, place, or scaffold on its own.
  assert.match(source, /movements\.canDig = false/)
  assert.match(source, /movements\.scafoldingBlocks = \[\]/)
  // The stone helpers never dig, chat, use creative mode or change what the pathfinder may do:
  // their stricter movement rules start from safeMovements (canDig false) and only tighten drops.
  const helpers = fs.readFileSync(require.resolve('./stone-helpers'), 'utf8')
  assert.doesNotMatch(helpers, /\.dig\s*\(|digBlock\s*\(/)
  assert.doesNotMatch(helpers, /\.chat\s*\(/)
  assert.doesNotMatch(helpers, /creative/i)
  assert.doesNotMatch(helpers, /canDig\s*=|scafoldingBlocks\s*=|allow1by1towers\s*=|setControlState|\.respawn\s*\(/)
  assert.match(helpers, /const movements = safeMovements\(bot, Movements, spawn\)\n  movements\.maxDropDown = 2\n/)
  // The bridge loads the stone helpers only inside functions, never when it is required.
  assert.doesNotMatch(source, /^(const|let|var)[^\n]*require\(['"]\.\/stone-helpers['"]\)/m)
})

// Step S1 of the stone stage (docs/stone_tools_design.md): goal gating, the
// stone observation fields, the passive survey and the stone recipe fill.
function cube (x0, x1, y0, y1, z0, z1, name) {
  const blocks = []
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) blocks.push([x, y, z, name])
  return blocks
}
// Open sky over a built region: full sky light in every empty cell above the
// first solid block of each column, as the server sends it for open ground.
function skyLit (world, x0, x1, z0, z1, top = 90) {
  for (let x = x0; x <= x1; x++) {
    for (let z = z0; z <= z1; z++) {
      for (let y = top; y > -64; y--) {
        const position = new Vec3(x, y, z)
        const block = world.getBlock(position)
        if (!block || block.boundingBox === 'block') break
        world.setSkyLight(position, 15)
      }
    }
  }
  return world
}
// The worked example of design section 4.4: grass, three dirt, then stone five blocks below the feet at y 64.
function workedExample (x0 = -8, x1 = 12, z0 = -8, z1 = 8) {
  return [...cube(x0, x1, 50, 59, z0, z1, 'stone'), ...cube(x0, x1, 60, 62, z0, z1, 'dirt'), ...layer(x0, x1, z0, z1, 63, 'grass_block')]
}
function stateIds (world, x0, x1, y0, y1, z0, z1) {
  const ids = []
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) ids.push(world.getBlockStateId(new Vec3(x, y, z)))
  return ids
}
const STONE_FIELD_KEYS = ['nearby_stone', 'stone_visible', 'stone_targets', 'pickaxe_uses_left', 'pickaxe_digs', 'hostile_nearby', 'time_of_day', 'depth_below_stage_start']

test('a stone pickaxe recipe is filled from the stone materials actually held, and the wooden recipes never use that fill', () => {
  const registry = fromMineflayer('prismarine-registry')('1.21.4')
  const Recipe = fromMineflayer('prismarine-recipe')(registry).Recipe
  const id = name => registry.itemsByName[name].id
  const templates = Recipe.find(id('stone_pickaxe'))
  const pristine = JSON.parse(JSON.stringify(templates))
  const table = { name: 'crafting_table', position: new Vec3(3, 64, 3) }
  let items = []
  const asked = []
  const bot = {
    registry,
    inventory: { items: () => items },
    recipesFor: () => [],
    recipesAll: (item, metadata, craftingTable) => { asked.push([item, metadata, craftingTable]); return templates }
  }
  const hold = list => { items = list.map(([name, count]) => ({ type: id(name), count })) }
  const materialRow = recipe => recipe.inShape.find(row => row.every(ingredient => ingredient.id !== -1 && ingredient.id !== id('stick'))).map(ingredient => registry.items[ingredient.id].name)
  // Mixed materials (decision 10): minecraft-data lists only single-material recipes, so recipesFor finds none.
  hold([['cobblestone', 2], ['cobbled_deepslate', 1], ['stick', 2]])
  const mixed = craftingRecipe(bot, registry.itemsByName.stone_pickaxe, table)
  assert.deepEqual(materialRow(mixed).sort(), ['cobbled_deepslate', 'cobblestone', 'cobblestone'])
  assert.equal(mixed.inShape.flat().filter(ingredient => ingredient.id === id('stick')).length, 2)
  assert.equal(mixed.requiresTable, true)
  assert.deepEqual(asked.at(-1), [id('stone_pickaxe'), null, table])
  hold([['blackstone', 2], ['cobblestone', 1], ['stick', 3]])
  assert.deepEqual(materialRow(craftingRecipe(bot, registry.itemsByName.stone_pickaxe, table)).sort(), ['blackstone', 'blackstone', 'cobblestone'])
  // Two materials, one stick, or stone-looking blocks that are not recipe materials: no recipe.
  for (const list of [[['cobblestone', 2], ['stick', 2]], [['cobblestone', 3], ['stick', 1]], [['stone', 3], ['granite', 3], ['mossy_cobblestone', 3], ['stick', 2]]]) {
    hold(list)
    assert.equal(craftingRecipe(bot, registry.itemsByName.stone_pickaxe, table), null, JSON.stringify(list))
  }
  assert.deepEqual(JSON.parse(JSON.stringify(templates)), pristine) // the library's templates are never changed
  // A recipe the library already finds is used as it is.
  const ordinary = { inShape: templates[0].inShape, requiresTable: true }
  assert.equal(craftingRecipe({ ...bot, recipesFor: () => [ordinary] }, registry.itemsByName.stone_pickaxe, table), ordinary)
  // The wooden recipes never reach the stone fill: with it replaced by a failing stub they behave as before.
  const helpers = require('./stone-helpers')
  const fill = helpers.fillStoneRecipe
  helpers.fillStoneRecipe = () => { throw new Error('the stone fill was used for a wooden recipe') }
  try {
    const plank = { id: 1, count: 1, metadata: null }
    const stick = { id: 3, count: 1, metadata: null }
    const empty = { id: -1, count: 1, metadata: null }
    const template = { inShape: [[plank, plank, plank], [empty, stick, empty], [empty, stick, empty]], requiresTable: true }
    const wood = { recipesFor: () => [], recipesAll: () => [template], registry: { items: { 1: { name: 'oak_planks' }, 2: { name: 'birch_planks' }, 3: { name: 'stick' } } }, inventory: { items: () => [{ type: 1, count: 1 }, { type: 2, count: 2 }, { type: 3, count: 2 }] } }
    assert.deepEqual(craftingRecipe(wood, { id: 7, name: 'wooden_pickaxe' }, {}).inShape[0].map(ingredient => ingredient.id), [1, 2, 2])
    assert.equal(craftingRecipe(wood, { id: 8, name: 'stone_axe' }, {}), null)
  } finally { helpers.fillStoneRecipe = fill }
})

test('the stone fields come from the reach list and the ray fan, count lit faces only, and stone in reach is always visible', async () => {
  const stone = require('./stone-helpers')
  const bot = new Vec3(0.5, 64, 0.5)
  const eyes = eyesAt(bot)
  const scene = (extra = {}) => ({ botPosition: bot, spawn: { x: 0.5, z: 0.5 }, table: null, entities: {}, startY: 64, ...extra })
  const ground = [...cube(-20, 20, 58, 62, -20, 20, 'stone'), ...layer(-20, 20, -20, 20, 63, 'grass_block')]
  const outcrop = [[2, 64, 0, 'stone']]
  const cliff = cube(10, 11, 64, 67, -3, 3, 'stone')
  const lit = async blocks => skyLit(await buildWorld(blocks), -20, 20, -20, 20)
  assert.equal(STONE_LIT_ONLY, true)

  const both = await lit([...ground, ...outcrop, ...cliff])
  const scan = stoneScan(both, scene())
  assert.deepEqual([scan.fields.nearby_stone, scan.fields.stone_visible], [true, true])
  assert.deepEqual(scan.fields.stone_targets, [{ name: 'stone', x: 2, y: 64, z: 0, distance: Math.round(new Vec3(2.5, 64.5, 0.5).distanceTo(eyes) * 100) / 100, drop: 'cobblestone', rest_y: 64 }])
  assert.deepEqual(scan.fields.hostile_nearby, [])
  // nearby_stone is exactly what MINE_STONE's reach function returns, with the same light rule.
  const reach = stone.reachableStone(both, stone.stoneContext(both, scene()), { litOnly: true })
  assert.deepEqual(scan.reach.map(candidate => `${candidate.position}`), reach.map(candidate => `${candidate.position}`))
  assert(scan.fan.targets.some(target => target.position.x === 10), 'the cliff is a SEARCH_STONE target')
  assert(Number.isFinite(scan.reach_ms) && Number.isFinite(scan.fan.stats.ms))

  const far = await lit([...ground, ...cliff])
  const seen = stoneScan(far, scene())
  assert.deepEqual([seen.fields.nearby_stone, seen.fields.stone_visible, seen.fields.stone_targets], [false, true, []])
  const buried = stoneScan(await lit(ground), scene())
  assert.deepEqual([buried.fields.nearby_stone, buried.fields.stone_visible], [false, false])
  // Decision 9: without light (a sealed cave) the same stone is neither seen nor in reach.
  const dark = await buildWorld([...ground, ...outcrop, ...cliff])
  assert.deepEqual([stoneScan(dark, scene()).fields.nearby_stone, stoneScan(dark, scene()).fields.stone_visible], [false, false])
  assert.deepEqual([stoneScan(dark, scene(), { litOnly: false }).fields.nearby_stone, stoneScan(dark, scene(), { litOnly: false }).fields.stone_visible], [true, true])

  // A zombie in sight within 12 blocks is listed, and every dig, so every target, is refused.
  const zombie = stoneScan(both, scene({ entities: { 7: { name: 'zombie', position: new Vec3(5.5, 64, 3.5), height: 1.95 } } }))
  assert.deepEqual(zombie.fields.hostile_nearby, [{ name: 'zombie', x: 5.5, y: 64, z: 3.5, distance: 5.83 }])
  assert.deepEqual([zombie.fields.nearby_stone, zombie.fields.stone_visible], [false, false])
  assert.deepEqual(stoneScan(both, scene({ entities: { 7: { name: 'cow', position: new Vec3(5.5, 64, 3.5), height: 1.4 } } })).fields.hostile_nearby, [])

  // stone_visible follows what SEARCH_STONE's next() would return: two failures in a row force exploration.
  const memory = new stone.StoneMemory()
  const targets = seen.fan.targets
  memory.failed(memory.next(targets).block)
  memory.failed(memory.next(targets).block)
  const blocked = stoneScan(far, scene(), { memory })
  assert(blocked.fan.targets.length > 0)
  assert.equal(blocked.fields.stone_visible, false)
  // Stone in reach is visible even when the fan has no target left for it.
  const single = await lit([...ground, ...outcrop])
  const forgetting = new stone.StoneMemory()
  forgetting.next(stoneScan(single, scene()).fan.targets)
  forgetting.failed(single.getBlock(new Vec3(2, 64, 0)))
  const inReach = stoneScan(single, scene(), { memory: forgetting })
  assert.deepEqual([inReach.fan.targets.length, inReach.fields.nearby_stone, inReach.fields.stone_visible], [0, true, true])
})

test('pickaxe uses left read the 1.21.4 damage component through prismarine-item, the held pickaxe first', () => {
  const registry = fromMineflayer('prismarine-registry')('1.21.4')
  const Item = fromMineflayer('prismarine-item')(registry)
  const protocol = fromMineflayer('minecraft-protocol')
  const serializer = protocol.createSerializer({ state: 'play', isServer: true, version: '1.21.4' })
  const deserializer = protocol.createDeserializer({ state: 'play', isServer: false, version: '1.21.4' })
  // A set_slot packet through the pinned codec, parsed as Mineflayer's inventory plugin parses it.
  const slot = (name, damage) => {
    const components = damage === undefined ? [] : [{ type: 'damage', data: damage }]
    const item = { itemCount: 1, itemId: registry.itemsByName[name].id, addedComponentCount: components.length, removedComponentCount: 0, components, removeComponents: [] }
    const params = deserializer.parsePacketBuffer(serializer.createPacketBuffer({ name: 'set_slot', params: { windowId: 0, stateId: 1, slot: 36, item } })).data.params
    return Item.fromNotch(params.item)
  }
  assert.equal(registry.itemsByName.wooden_pickaxe.maxDurability, 59)
  const fresh = slot('wooden_pickaxe')
  assert.deepEqual([fresh.durabilityUsed, fresh.maxDurability], [0, 59])
  assert.equal(pickaxeUsesLeft([fresh], fresh), 59)
  const used = slot('wooden_pickaxe', 1)
  assert.equal(used.durabilityUsed, 1)
  assert.equal(pickaxeUsesLeft([used]), 58)
  const worn = slot('wooden_pickaxe', 58)
  assert.equal(pickaxeUsesLeft([worn, used], used), 58)
  assert.equal(pickaxeUsesLeft([worn, used], slot('stone_pickaxe', 3)), 1) // the wooden pickaxe, not the held stone one
  assert.equal(pickaxeUsesLeft([slot('stone_pickaxe')]), null)
  assert.equal(pickaxeUsesLeft([]), null)
  assert.equal(pickaxeUsesLeft([{ name: 'wooden_pickaxe', maxDurability: 59, get durabilityUsed () { throw new Error('unknown durability location') } }]), null)
  assert.equal(pickaxeUsesLeft([{ name: 'wooden_pickaxe', maxDurability: 59, durabilityUsed: null }]), null)
})

test('the survey reports the depth to stone and four simulated staircases without digging anything', async (t) => {
  const stone = require('./stone-helpers')
  const { Movements } = require('mineflayer-pathfinder')
  const Move = require('mineflayer-pathfinder/lib/move')
  const spawn = { x: 0.5, z: 0.5 }
  const reversible = stone.stairReversibility(fake => stone.strictMovements(fake, Movements, spawn), Move)
  const scene = (world, extra = {}) => ({ botPosition: new Vec3(0.5, 64, 0.5), spawn, table: null, entities: {}, startY: 64, ...extra })
  const feet = new Vec3(0, 64, 0)
  const world = skyLit(await buildWorld(workedExample()), -8, 12, -8, 8)
  assert.deepEqual(depthToStone(world, feet), { depth: 5, block: 'stone', layers: [['grass_block', 1], ['dirt', 3]], stopped: null })
  assert.deepEqual(depthToStone(world, feet, 4), { depth: null, block: null, layers: [['grass_block', 1], ['dirt', 3]], stopped: 'scan_limit' })
  assert.equal(depthToStone(world, new Vec3(300, 64, 0)).stopped, 'unloaded')

  const before = stateIds(world, -8, 12, 50, 66, -8, 8)
  const scan = stoneScan(world, scene(world))
  assert.deepEqual([scan.fields.nearby_stone, scan.fields.stone_visible], [false, false]) // all stone is buried
  const record = stoneSurveyRecord(world, feet, scan, { startY: 64, reversible })
  assert.deepEqual(stateIds(world, -8, 12, 50, 66, -8, 8), before) // nothing was dug
  assert.deepStrictEqual(JSON.parse(JSON.stringify(record)), record) // plain data only, no blocks or vectors
  // Flat ground: every heading digs the worked example's 1, 2, 3, 3 and 2 cells by hand and exposes stone level with the floor.
  assert.deepEqual(Object.keys(record.staircases), ['+x', '+z', '-x', '-z'])
  for (const stair of Object.values(record.staircases)) assert.deepEqual([stair.stopped, stair.dig_counts, stair.end_y, stair.steps.length], ['stone_exposed', [1, 2, 3, 3, 2], 60, 5])
  assert.deepEqual(record.staircases['+x'].steps[0], { ok: true, digs: [['grass_block', 'hand']], dig_ms: 900, move: true, landing_y: 63, floor: 'dirt' })
  assert.deepEqual(record.staircases['+x'].steps[4], { ok: true, digs: [['dirt', 'hand'], ['dirt', 'hand']], dig_ms: 1500, move: false, exposes: 'stone' })
  // A planned dig is never a stone source.
  const dug = Object.values(record.staircases).flatMap(stair => stair.steps.flatMap(step => step.digs || []))
  assert(dug.length === 44 && dug.every(([name]) => !stone.isStoneSource(name)))
  assert.deepEqual(record.depth_to_stone, depthToStone(world, feet))
  assert.equal(record.lit_only, true)
  assert.deepEqual([record.fan.rays, record.fan.targets, record.fan.targets_any_light, record.reach.candidates, record.reach.candidates_any_light], [416, 0, 0, 0, 0])
  assert(record.fan.ms >= 0 && record.reach.ms >= 0 && record.survey_ms >= 0)
  t.diagnostic(`survey on the worked example: fan ${record.fan.ms} ms (${record.fan.reads} block reads), reach ${record.reach.ms} ms, survey ${record.survey_ms} ms`)

  // A river along z = 1 refuses the headings beside it on their first step; a zombie in sight refuses all four.
  const river = skyLit(await buildWorld([...workedExample(), ...cube(-8, 12, 62, 63, 1, 1, 'water')]), -8, 12, -8, 8)
  const wet = surveyStaircases(river, feet, { ...stoneScan(river, scene(river)).ctx, startY: 64, reversible })
  assert.deepEqual(Object.fromEntries(Object.entries(wet).map(([heading, stair]) => [heading, stair.steps[0].reason || null])),
    { '+x': 'liquid_adjacent', '+z': 'liquid_adjacent', '-x': 'liquid_adjacent', '-z': null })
  assert.equal(wet['+z'].steps[0].ok, false)
  const watched = stoneScan(world, scene(world, { entities: { 3: { name: 'skeleton', position: new Vec3(-4.5, 64, 4.5), height: 1.99 } } }))
  assert.equal(watched.fields.hostile_nearby.length, 1)
  const refused = surveyStaircases(world, feet, { ...watched.ctx, startY: 64, reversible })
  for (const stair of Object.values(refused)) assert.deepEqual([stair.stopped, stair.steps.length, stair.dig_counts], ['hostile_nearby', 1, []])
  // Rock that is not a recipe material is dug with the pickaxe; the depth limit counts from the stage start
  // (a start nine blocks above these feet leaves room for three steps).
  const granite = skyLit(await buildWorld([...workedExample(), [1, 63, 0, 'granite']]), -8, 12, -8, 8)
  assert.deepEqual(surveyStaircases(granite, feet, { ...stoneScan(granite, scene(granite)).ctx, startY: 64, reversible }, 1)['+x'].steps[0].digs, [['granite', 'pickaxe']])
  const deep = surveyStaircases(world, feet, { ...scan.ctx, startY: 64 + 9, reversible })
  assert.deepEqual([deep['+x'].stopped, deep['+x'].dig_counts, deep['+x'].end_y], ['depth_limit', [1, 2, 3], 61])
  assert.equal(SURVEY_STAIR_STEPS, 6)
})

// The real bridge process with Mineflayer replaced by a stand-in bot on an
// offline world (node --require), so goal gating and the observation schema
// are tested on main() itself. The stand-in never moves, digs or crafts; its
// inventory fills only after the bridge has checked that it starts empty. It
// counts calls to the stone helpers' scans and can make one of them fail.
function fakeMineflayer () {
  const Module = require('node:module')
  const fs = require('node:fs')
  const { EventEmitter } = require('node:events')
  const scenario = JSON.parse(fs.readFileSync(process.env.FAKE_BOT_SCENARIO, 'utf8'))
  const calls = {}
  let wrapped = false
  process.on('exit', () => fs.writeFileSync(process.env.FAKE_BOT_CALLS, JSON.stringify(calls)))
  const load = Module._load
  Module._load = function (request, parent) {
    if (request === 'mineflayer' && /resource-bridge\.js$/.test(parent?.filename || '')) return { createBot }
    const loaded = load.apply(this, arguments)
    if (request === './stone-helpers' && !wrapped) {
      wrapped = true
      for (const name of ['stoneInventoryCounts', 'stoneContext', 'reachableStone', 'visibleStoneTargets', 'planStaircase', 'fillStoneRecipe']) {
        const original = loaded[name]
        loaded[name] = (...args) => {
          calls[name] = (calls[name] || 0) + 1
          if (process.env.FAKE_BOT_FAIL === name) throw new Error(`injected ${name} failure`)
          return original(...args)
        }
      }
    }
    return loaded
  }
  function createBot (options) {
    calls.createBot = options
    const bridge = process.env.FAKE_BOT_BRIDGE
    const mineflayerEntry = require.resolve('mineflayer', { paths: [bridge] })
    const from = name => require(require.resolve(name, { paths: [mineflayerEntry] }))
    const { Vec3 } = require(require.resolve('vec3', { paths: [bridge] }))
    const registry = from('prismarine-registry')('1.21.4')
    const World = from('prismarine-world')('1.21.4')
    const Chunk = from('prismarine-chunk')('1.21.4')
    const Item = from('prismarine-item')(registry)
    const [x0, x1, y0, y1, z0, z1] = scenario.bounds
    const bot = new EventEmitter()
    bot._client = new EventEmitter()
    let started = false
    let items = []
    Object.assign(bot, {
      registry,
      version: '1.21.4',
      game: { gameMode: 'survival', minY: -64, height: 384 },
      health: 20,
      food: 20,
      players: {},
      entities: {},
      time: { timeOfDay: scenario.timeOfDay },
      inventory: { items: () => started ? items : [] },
      loadPlugin () {
        bot.pathfinder = {
          setMovements (movements) { bot.pathfinder.movements = movements; started = true },
          setGoal () {},
          goto: () => Promise.reject(new Error('the stand-in bot never moves'))
        }
      },
      clearControlStates () {},
      stopDigging () {},
      quit () {},
      waitForChunksToLoad: async () => {},
      recipesFor: () => [],
      recipesAll: () => [],
      canDigBlock: () => true,
      nearestEntity: () => null,
      blockAt: position => bot.world.getBlock(position),
      findBlocks ({ matching, maxDistance, count }) {
        const origin = bot.entity.position
        const found = []
        for (let x = x0; x <= x1; x++) {
          for (let y = y0; y <= y1; y++) {
            for (let z = z0; z <= z1; z++) {
              const position = new Vec3(x, y, z)
              if (position.distanceTo(origin) <= maxDistance && matching(bot.world.getBlock(position))) found.push(position)
            }
          }
        }
        return found.sort((a, b) => a.distanceTo(origin) - b.distanceTo(origin)).slice(0, count)
      }
    })
    Object.defineProperty(bot, 'heldItem', { get: () => (started && items.find(item => item.name === scenario.held)) || null })
    setImmediate(async () => {
      const world = new World(null)
      for (let cx = Math.floor(x0 / 16) - 1; cx <= Math.floor(x1 / 16) + 1; cx++) {
        for (let cz = Math.floor(z0 / 16) - 1; cz <= Math.floor(z1 / 16) + 1; cz++) await world.setColumn(cx, cz, new Chunk({ minY: -64, worldHeight: 384 }))
      }
      for (const [bx0, bx1, by0, by1, bz0, bz1, name] of scenario.boxes) {
        const state = registry.blocksByName[name].defaultState
        for (let x = bx0; x <= bx1; x++) for (let y = by0; y <= by1; y++) for (let z = bz0; z <= bz1; z++) world.sync.setBlockStateId(new Vec3(x, y, z), state)
      }
      for (let x = x0; x <= x1; x++) {
        for (let z = z0; z <= z1; z++) {
          for (let y = y1 + 2; y > -64 && world.sync.getBlock(new Vec3(x, y, z)).boundingBox === 'empty'; y--) world.sync.setSkyLight(new Vec3(x, y, z), 15)
        }
      }
      bot.world = world.sync
      bot.entity = { position: new Vec3(...scenario.position), onGround: true, height: 1.8, width: 0.6, name: 'player', effects: {} }
      bot.entities = { 0: bot.entity }
      items = scenario.inventory.map(([name, count, damage]) => Item.fromNotch({
        itemId: registry.itemsByName[name].id, itemCount: count, addedComponentCount: damage ? 1 : 0, removedComponentCount: 0,
        components: damage ? [{ type: 'damage', data: damage }] : [], removeComponents: []
      }))
      bot.emit('spawn')
      bot._client.emit('window_items', { windowId: 0 })
    })
    return bot
  }
}

// Run a bridge file under the stand-in with JSON-lines requests on stdin; resolves with every response line.
function runBridge ({ args = [], scenario, requests, bridge = require.resolve('./resource-bridge'), fail = null }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'resource-bridge-stand-in-'))
  const files = { preload: path.join(dir, 'stand-in.js'), scenario: path.join(dir, 'scenario.json'), calls: path.join(dir, 'calls.json') }
  fs.writeFileSync(files.preload, `'use strict'\n;(${fakeMineflayer})()\n`)
  fs.writeFileSync(files.scenario, JSON.stringify(scenario))
  const env = {
    ...process.env,
    NODE_PATH: [process.env.NODE_PATH, path.join(__dirname, 'node_modules')].filter(Boolean).join(path.delimiter),
    FAKE_BOT_SCENARIO: files.scenario,
    FAKE_BOT_CALLS: files.calls,
    FAKE_BOT_BRIDGE: __dirname
  }
  delete env.FAKE_BOT_FAIL
  if (fail) env.FAKE_BOT_FAIL = fail
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--require', files.preload, bridge, ...args], { cwd: path.dirname(bridge), env })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', chunk => { stdout += chunk })
    child.stderr.on('data', chunk => { stderr += chunk })
    child.stdin.on('error', () => {})
    const timer = setTimeout(() => child.kill(), 60000)
    child.on('error', reject)
    child.on('close', code => {
      clearTimeout(timer)
      try {
        const calls = fs.existsSync(files.calls) ? JSON.parse(fs.readFileSync(files.calls, 'utf8')) : null
        resolve({ code, stderr, calls, responses: stdout.split('\n').filter(Boolean).map(line => JSON.parse(line)) })
      } catch (error) { reject(error) } finally { fs.rmSync(dir, { recursive: true, force: true }) }
    })
    child.stdin.end(requests.map((request, index) => JSON.stringify({ id: index + 1, ...request })).join('\n') + '\n')
  })
}
// Grass on stone, an exposed stone block in reach, a stone cliff in sight and an oak trunk out of reach, under open sky.
const STAND_IN_WORLD = {
  bounds: [-12, 12, 56, 70, -12, 12],
  boxes: [[-12, 12, 56, 62, -12, 12, 'stone'], [-12, 12, 63, 63, -12, 12, 'grass_block'], [2, 2, 64, 64, 0, 0, 'stone'],
    [8, 9, 64, 66, -3, 3, 'stone'], [4, 4, 64, 67, -6, -6, 'oak_log']],
  position: [0.5, 64, 0.5],
  timeOfDay: 1000
}
const standIn = (inventory = [], held = null) => ({ ...STAND_IN_WORLD, inventory, held })
// Two uses already spent on the wooden pickaxe.
const WOODEN_HELD = standIn([['wooden_pickaxe', 1, 2], ['stick', 2], ['oak_planks', 1]], 'wooden_pickaxe')
// The observation keys of the wood stage at 409560e, in order.
const WOOD_OBSERVATION_KEYS = ['position', 'health', 'food', 'gamemode', 'inventory', 'inventory_counts', 'nearby_log', 'nearby_table', 'table_placed',
  'nearby_visible_logs', 'biome', 'feet_block', 'head_block', 'ground_block', 'support_block', 'on_canopy', 'adjacent_drops', 'table_position',
  'held_item', 'success', 'death', 'recovery', 'action_detail', 'event', 'action_duration_seconds', 'backend', 'game_version']
// The wood-stage bridge before step S1 (unchanged since 529bf7d).
const WOOD_BASELINE = '409560e'
const STONE_SCANS = ['stoneContext', 'reachableStone', 'visibleStoneTargets', 'planStaircase']
// A response with the step's measured duration removed, after checking it is a number.
function withoutDuration (response) {
  if (!response.ok || response.result.event === 'already_completed' || !('action_duration_seconds' in response.result)) return response
  assert.equal(typeof response.result.action_duration_seconds, 'number')
  const { action_duration_seconds: _, ...result } = response.result
  return { ...response, result }
}

test('under the default goal the bridge answers exactly as the 409560e bridge did, apart from the two stone counts', async (t) => {
  const shown = spawnSync('git', ['show', `${WOOD_BASELINE}:minecraft_bridge/resource-bridge.js`], { cwd: __dirname, encoding: 'utf8' })
  if (shown.status !== 0) { t.skip(`git history is not available here: ${(shown.stderr || shown.error?.message || '').trim()}`); return }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'resource-bridge-baseline-'))
  const baseline = path.join(dir, 'resource-bridge.js')
  fs.writeFileSync(baseline, shown.stdout)
  try {
    const requests = [{ op: 'status' }, { op: 'observe' }, { op: 'step', action: 1 }, { op: 'step', action: 0 }, { op: 'step', action: 8 }, { op: 'observe' }]
    for (const [label, scenario] of [['empty inventory', standIn()], ['wooden pickaxe held', WOODEN_HELD]]) {
      const [now, explicit, before] = await Promise.all([
        runBridge({ scenario, requests }),
        runBridge({ scenario, requests, args: ['--goal', 'wooden_pickaxe', '--stone-survey', 'off'] }),
        runBridge({ scenario, requests, bridge: baseline })
      ])
      // Five answers each: the refused action 8 ends the session, so the last request is never answered.
      assert.deepEqual([now.responses.length, before.responses.length], [5, 5], `${label}: ${now.stderr} ${before.stderr}`)
      assert.deepEqual(before.responses[4], { id: 5, ok: false, error: 'action must be an integer from 0 through 7' })
      assert.deepEqual(explicit.responses.map(withoutDuration), now.responses.map(withoutDuration), label)
      const stripped = now.responses.map(response => {
        if (!response.ok || !response.result.inventory_counts) return response
        const { stone_materials: materials, stone_pickaxes: pickaxes, ...counts } = response.result.inventory_counts
        assert.deepEqual([materials, pickaxes], [0, 0])
        return { ...response, result: { ...response.result, inventory_counts: counts } }
      })
      assert.deepEqual(stripped.map(withoutDuration), before.responses.map(withoutDuration), label)
      // The bot connects with the same options: the goal and the survey never reach Mineflayer.
      assert.deepEqual(now.calls.createBot, before.calls.createBot)
      assert.deepEqual(explicit.calls.createBot, { host: '127.0.0.1', port: 25567, username: 'RLResourceBot', version: '1.21.4', auth: 'offline', hideErrors: true, respawn: false })
      // The same keys in the same order as before, and not one stone scan.
      for (const response of now.responses.slice(1, 4)) assert.deepEqual(Object.keys(response.result), WOOD_OBSERVATION_KEYS, label)
      for (const name of STONE_SCANS) assert.equal(now.calls[name], undefined, `${label}: ${name} ran under the default goal`)
      assert(now.calls.stoneInventoryCounts > 0)
    }
  } finally { fs.rmSync(dir, { recursive: true, force: true }) }
})

test('under the default goal a wooden pickaxe completes the episode and actions 8 to 12 end the session as before', async () => {
  const runs = await Promise.all([8, 9, 10, 11, 12].map(action => runBridge({ scenario: WOODEN_HELD, requests: [{ op: 'step', action: 0 }, { op: 'step', action }, { op: 'observe' }] })))
  for (const [index, run] of runs.entries()) {
    assert.equal(run.responses.length, 2, run.stderr)
    const [completed, refused] = run.responses
    assert.deepEqual([completed.result.event, completed.result.success, completed.result.action_duration_seconds], ['already_completed', true, 0])
    assert.deepEqual(refused, { id: 2, ok: false, error: 'action must be an integer from 0 through 7' }, `action ${index + 8}`)
    assert.equal(run.code, 0)
  }
})

test('under the stone goal the fields are null until a wooden pickaxe, success needs a held stone pickaxe, and stone actions fail loudly', async () => {
  const args = ['--goal', 'stone_pickaxe']
  const crafted = [['wooden_pickaxe', 1], ['stone_pickaxe', 1], ['oak_planks', 1]]
  const [empty, wooden, done, unheld] = await Promise.all([
    runBridge({ args, scenario: standIn(), requests: [{ op: 'observe' }, { op: 'step', action: 8 }, { op: 'observe' }] }),
    runBridge({ args, scenario: WOODEN_HELD, requests: [{ op: 'observe' }, { op: 'step', action: 1 }, { op: 'step', action: 13 }] }),
    runBridge({ args, scenario: standIn(crafted, 'stone_pickaxe'), requests: [{ op: 'observe' }, { op: 'step', action: 0 }, { op: 'step', action: 12 }] }),
    runBridge({ args, scenario: standIn(crafted, 'wooden_pickaxe'), requests: [{ op: 'observe' }] })
  ])
  // Before any wooden pickaxe: every stone field is present and null, and nothing is scanned.
  const start = empty.responses[0].result
  assert.deepEqual(Object.keys(start), [...WOOD_OBSERVATION_KEYS, ...STONE_FIELD_KEYS])
  assert(STONE_FIELD_KEYS.every(key => start[key] === null))
  assert.equal(start.success, false)
  for (const name of STONE_SCANS) assert.equal(empty.calls[name], undefined)
  assert.deepEqual(empty.responses.slice(1), [{ id: 2, ok: false, error: 'SEARCH_STONE (action 8) is not implemented until step S3; this episode is invalid' }])
  // With the wooden pickaxe held the stage has started: the fields are measured, and the wood actions still run.
  const handoff = wooden.responses[0].result
  assert.equal(handoff.success, false)
  assert.deepEqual(handoff.inventory_counts, { logs: 0, planks: 1, sticks: 2, tables: 0, pickaxes: 1, stone_materials: 0, stone_pickaxes: 0 })
  assert.deepEqual([handoff.nearby_stone, handoff.stone_visible, handoff.pickaxe_uses_left, handoff.pickaxe_digs, handoff.time_of_day, handoff.depth_below_stage_start],
    [true, true, 57, 0, 1000, 0])
  assert.deepEqual(handoff.stone_targets.map(target => [target.name, target.x, target.y, target.z, target.drop, target.rest_y]), [['stone', 2, 64, 0, 'cobblestone', 64]])
  assert.deepEqual(handoff.hostile_nearby, [])
  assert.equal('diagnostic_only' in handoff, false)
  assert.deepEqual([wooden.responses[1].ok, wooden.responses[1].result.event, wooden.responses[1].result.success], [true, 'no_reachable_log', false])
  assert.deepEqual(wooden.responses[2], { id: 3, ok: false, error: 'action must be an integer from 0 through 12' })
  assert(STONE_SCANS.slice(0, 3).every(name => wooden.calls[name] >= 2))
  // Only a held stone pickaxe completes the stone goal.
  assert.equal(done.responses[0].result.success, true)
  assert.deepEqual([done.responses[1].result.event, done.responses[1].result.action_duration_seconds], ['already_completed', 0])
  assert.deepEqual(done.responses[2], { id: 3, ok: false, error: 'CRAFT_STONE_PICKAXE (action 12) is not implemented until step S3; this episode is invalid' })
  assert.deepEqual([unheld.responses[0].result.success, unheld.responses[0].result.inventory_counts.stone_pickaxes], [false, 1])
  assert.deepEqual(Object.keys(unheld.calls.createBot), ['host', 'port', 'username', 'version', 'auth', 'hideErrors', 'respawn'])
})

test('the passive survey adds diagnostic_only once a wooden pickaxe is held and never changes the wood outcome', async () => {
  const args = ['--stone-survey', 'on']
  const [empty, wooden, failing, stoneFailing] = await Promise.all([
    runBridge({ args, scenario: standIn(), requests: [{ op: 'observe' }] }),
    runBridge({ args, scenario: WOODEN_HELD, requests: [{ op: 'observe' }, { op: 'step', action: 0 }] }),
    runBridge({ args, scenario: WOODEN_HELD, requests: [{ op: 'observe' }], fail: 'planStaircase' }),
    runBridge({ args: ['--goal', 'stone_pickaxe', ...args], scenario: WOODEN_HELD, requests: [{ op: 'observe' }], fail: 'planStaircase' })
  ])
  const start = empty.responses[0].result
  assert.deepEqual(Object.keys(start), [...WOOD_OBSERVATION_KEYS, ...STONE_FIELD_KEYS, 'diagnostic_only'])
  assert.equal(start.diagnostic_only, null)
  for (const name of STONE_SCANS) assert.equal(empty.calls[name], undefined)
  // The hand-off observation of a wood run: still a success, now with the stone fields and the survey record.
  const handoff = wooden.responses[0].result
  assert.deepEqual([handoff.success, handoff.nearby_stone, handoff.stone_visible, handoff.pickaxe_uses_left], [true, true, true, 57])
  const survey = handoff.diagnostic_only
  assert.deepEqual(Object.keys(survey), ['lit_only', 'depth_to_stone', 'staircases', 'fan', 'reach', 'survey_ms'])
  assert.deepEqual(survey.depth_to_stone, { depth: 2, block: 'stone', layers: [['grass_block', 1]], stopped: null })
  assert.deepEqual(Object.keys(survey.staircases), ['+x', '+z', '-x', '-z'])
  assert(Object.values(survey.staircases).every(stair => typeof stair.stopped === 'string' && stair.steps.length >= 1 && stair.steps.length <= 6))
  assert.deepEqual([survey.fan.rays, survey.reach.candidates, survey.lit_only], [416, 1, true])
  assert(survey.fan.targets >= 1 && Number.isFinite(survey.fan.ms))
  // Four headings for each of three observations: the bridge's own first look (the stand-in's inventory is
  // already full then), the observe request, and the step's look before acting.
  assert.equal(wooden.calls.planStaircase, 12)
  assert.deepEqual([wooden.responses[1].result.event, wooden.responses[1].result.diagnostic_only.depth_to_stone], ['already_completed', survey.depth_to_stone])
  // A survey failure under the wood goal is recorded and the wood observation still arrives.
  const recorded = failing.responses[0]
  assert.deepEqual([recorded.ok, recorded.result.success, recorded.result.nearby_stone, recorded.result.diagnostic_only], [true, true, true, { error: 'injected planStaircase failure' }])
  assert.match(failing.stderr, /Stone survey failed: injected planStaircase failure/)
  // Under the stone goal the same failure ends the session: the stone state needs these fields.
  assert.deepEqual(stoneFailing.responses, [{ id: 1, ok: false, error: 'injected planStaircase failure' }])
})
