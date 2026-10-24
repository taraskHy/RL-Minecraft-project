'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { spawnSync } = require('node:child_process')
const { EventEmitter } = require('node:events')
const { Vec3 } = require('vec3')
const {
  STONE_SOURCES, STONE_LOOKALIKES, HAND_DIG_BLOCKS, PICKAXE_DIG_BLOCKS, WOODEN_PICKAXE, DIG_REFUSALS, FAN_ELEVATIONS, MAX_STAIR_DEPTH,
  SPAWNER_BLOCKS, isStoneSource, isStoneMaterial, stoneDrop, isInfested, isGravityBlock, digClass, stoneInventoryCounts, chooseTool,
  inHitboxColumns, openFaces, visibleHostiles, findBlocksByName, stoneContext, digVerdict, dropRestY, inDropBand, reachableStone, rayFan,
  visibleStoneTargets, StoneMemory, clearedView, planStairStep, chooseStairHeading, planStaircase, strictMovements, stairReversibility,
  fillStoneRecipe
} = require('./stone-helpers')
const { SearchMemory, MAX_RADIUS, TARGET_MARGIN } = require('./resource-bridge')

// The offline 1.21.4 world builder of resource-bridge.test.js.
const mineflayerEntry = require.resolve('mineflayer')
const fromMineflayer = name => require(require.resolve(name, { paths: [mineflayerEntry] }))
const registry = fromMineflayer('prismarine-registry')('1.21.4')
async function buildWorld (blocks) {
  const version = '1.21.4'
  const World = fromMineflayer('prismarine-world')(version)
  const Chunk = fromMineflayer('prismarine-chunk')(version)
  const Block = fromMineflayer('prismarine-block')(version)
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
    const state = properties ? Block.fromProperties(name, properties, 0).stateId : registry.blocksByName[name].defaultState
    world.sync.setBlockStateId(new Vec3(x, y, z), state)
  }
  return world.sync
}
function box (x0, x1, y0, y1, z0, z1, name) {
  const blocks = []
  for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) blocks.push([x, y, z, name])
  return blocks
}
// Later entries replace earlier ones at the same cell.
function terrain (...parts) {
  const cells = new Map()
  for (const block of parts.flat()) cells.set(`${block[0]},${block[1]},${block[2]}`, block)
  return [...cells.values()]
}
// The worked example of section 4.4: grass, three dirt, then stone five blocks below the feet at y 64.
function workedExample (x0 = -6, x1 = 12, z0 = -6, z1 = 6) {
  return [...box(x0, x1, 50, 59, z0, z1, 'stone'), ...box(x0, x1, 60, 62, z0, z1, 'dirt'), ...box(x0, x1, 63, 63, z0, z1, 'grass_block')]
}
const at = (x, y, z) => new Vec3(x, y, z)
const centre = feet => new Vec3(feet.x + 0.5, feet.y, feet.z + 0.5)
const spawn = { x: 0.5, z: 0.5 }
function context (world, { bot = at(0.5, 64, 0.5), table = null, entities = [], startY = 64, home = spawn } = {}) {
  return stoneContext(world, { botPosition: bot, spawn: home, table, entities, startY })
}
const { Movements } = require('mineflayer-pathfinder')
const Move = require('mineflayer-pathfinder/lib/move')
const reversible = stairReversibility(bot => strictMovements(bot, Movements, spawn), Move)
function stairContext (world, options = {}) {
  return { ...context(world, options), reversible }
}
const reachStop = ctx => (view, feet) => reachableStone(view, { ...ctx, botPosition: centre(feet) }).length > 0

test('stone sources, their drops and lookalikes are recognized by exact name', () => {
  assert.deepEqual(STONE_SOURCES, ['stone', 'cobblestone', 'deepslate', 'cobbled_deepslate'])
  assert.deepEqual(STONE_SOURCES.map(stoneDrop), ['cobblestone', 'cobblestone', 'cobbled_deepslate', 'cobbled_deepslate'])
  assert.equal(stoneDrop('blackstone'), 'blackstone')
  assert.equal(isStoneSource('blackstone'), false) // counted in the inventory, never searched for
  for (const name of [...STONE_LOOKALIKES, 'red_sandstone', 'smooth_stone', 'stone_bricks', 'infested_stone', 'stone_slab']) {
    assert.equal(isStoneSource(name), false, name)
    assert.equal(stoneDrop(name), null, name)
  }
  assert.deepEqual(STONE_LOOKALIKES, ['granite', 'andesite', 'diorite', 'tuff', 'calcite', 'sandstone', 'terracotta', 'mossy_cobblestone'])
  for (const name of ['infested_stone', 'infested_cobblestone', 'infested_deepslate', 'infested_stone_bricks']) assert.equal(isInfested(name), true)
  assert.equal(isInfested('stone'), false)
  const falling = ['sand', 'red_sand', 'gravel', 'suspicious_sand', 'suspicious_gravel', 'white_concrete_powder', 'black_concrete_powder', 'anvil', 'damaged_anvil']
  for (const name of falling) assert.equal(isGravityBlock(name), true, name)
  for (const name of ['sandstone', 'white_concrete', 'dirt', 'pointed_dripstone']) assert.equal(isGravityBlock(name), false, name)
  // What DIG_DOWN digs, and with what; stone sources, logs, leaves, ores and ice are never dug.
  for (const name of HAND_DIG_BLOCKS) assert.equal(digClass(name), 'hand', name)
  for (const name of PICKAXE_DIG_BLOCKS) assert.equal(digClass(name), 'pickaxe', name)
  for (const name of [...STONE_SOURCES, 'oak_log', 'oak_leaves', 'coal_ore', 'iron_ore', 'ice', 'chest', 'crafting_table', 'mossy_cobblestone']) assert.equal(digClass(name), null, name)
})

test('stone materials are counted across cobblestone, cobbled deepslate and blackstone', () => {
  assert.deepEqual(stoneInventoryCounts({ cobblestone: 2, cobbled_deepslate: 1, blackstone: 4, stone: 9, granite: 3, stone_pickaxe: 1, wooden_pickaxe: 1 }), { stone_materials: 7, stone_pickaxes: 1 })
  assert.deepEqual(stoneInventoryCounts({}), { stone_materials: 0, stone_pickaxes: 0 })
  for (const name of ['cobblestone', 'cobbled_deepslate', 'blackstone']) assert.equal(isStoneMaterial(name), true)
  for (const name of ['stone', 'deepslate', 'mossy_cobblestone']) assert.equal(isStoneMaterial(name), false)
  for (const amount of [-1, 0.5, NaN, '2']) assert.throws(() => stoneInventoryCounts({ cobblestone: amount }))
})

test('the pickaxe is used on pickaxe blocks and the hand otherwise; ores, obsidian and slow digs are refused', async () => {
  const names = ['stone', 'cobblestone', 'deepslate', 'granite', 'terracotta', 'dirt', 'grass_block', 'sand', 'gravel', 'clay', 'iron_ore', 'copper_ore',
    'obsidian', 'coal_ore', 'ender_chest', 'oak_door', 'oak_log', 'oak_leaves', 'snow_block']
  const world = await buildWorld(names.map((name, i) => [i, 64, 0, name]))
  const tool = name => chooseTool(world.getBlock(at(names.indexOf(name), 64, 0)))
  for (const name of ['stone', 'cobblestone', 'deepslate', 'granite', 'terracotta', 'iron_ore', 'obsidian', 'coal_ore', 'ender_chest']) {
    assert.equal(tool(name).tool, 'pickaxe', name)
    assert.equal(tool(name).type, WOODEN_PICKAXE)
  }
  for (const name of ['dirt', 'grass_block', 'sand', 'gravel', 'clay', 'oak_door', 'oak_log', 'oak_leaves', 'snow_block']) {
    assert.equal(tool(name).tool, 'hand', name)
    assert.equal(tool(name).type, null)
  }
  assert.deepEqual([tool('stone').ms, tool('dirt').ms, tool('grass_block').ms, tool('granite').ms], [1150, 750, 900, 1150])
  // Harvest tools that exclude the wooden pickaxe: iron and copper ore, obsidian.
  for (const name of ['iron_ore', 'copper_ore', 'obsidian']) assert.equal(tool(name).harvestable, false, name)
  for (const name of ['stone', 'coal_ore', 'snow_block', 'dirt']) assert.equal(tool(name).harvestable, true, name)
  const ctx = context(world, { bot: at(0.5, 70, 10.5) })
  for (const name of ['iron_ore', 'copper_ore', 'obsidian']) assert.equal(digVerdict(world, at(names.indexOf(name), 64, 0), ctx).reason, 'wrong_tool', name)
  // Over 4 s with the chosen tool: an ender chest takes 16.9 s with the pickaxe and a door 4.55 s by hand.
  assert.deepEqual([tool('ender_chest').ms, tool('oak_door').ms], [16900, 4550])
  for (const name of ['ender_chest', 'oak_door']) assert.equal(digVerdict(world, at(names.indexOf(name), 64, 0), ctx).reason, 'wrong_tool', name)
  const stone = digVerdict(world, at(0, 64, 0), ctx)
  assert.deepEqual([stone.ok, stone.tool, stone.type, stone.ms], [true, 'pickaxe', WOODEN_PICKAXE, 1150])
})

test('digVerdict refuses a liquid on each of the six faces, a waterlogged neighbour and a waterlogged cell', async () => {
  const base = box(-3, 3, 60, 63, -3, 3, 'stone')
  const target = at(0, 61, 0)
  const ok = await buildWorld(base)
  assert.equal(digVerdict(ok, target, context(ok, { bot: at(0.5, 70, 5.5) })).ok, true)
  const faces = { west: [-1, 0, 0], east: [1, 0, 0], bottom: [0, -1, 0], top: [0, 1, 0], north: [0, 0, -1], south: [0, 0, 1] }
  for (const [face, [dx, dy, dz]] of Object.entries(faces)) {
    for (const liquid of ['water', 'lava']) {
      const world = await buildWorld(terrain(base, [[dx, 61 + dy, dz, liquid]]))
      const verdict = digVerdict(world, target, context(world, { bot: at(0.5, 70, 5.5) }))
      assert.deepEqual([verdict.reason, verdict.face, verdict.neighbour?.name], ['liquid_adjacent', face, liquid], `${liquid} on ${face}`)
    }
  }
  for (const liquid of ['bubble_column', 'seagrass', 'kelp', 'kelp_plant', 'tall_seagrass']) {
    const world = await buildWorld(terrain(base, [[1, 61, 0, liquid]]))
    assert.equal(digVerdict(world, target, context(world, { bot: at(0.5, 70, 5.5) })).reason, 'liquid_adjacent', liquid)
  }
  const logged = await buildWorld(terrain(base, [[0, 61, 1, 'oak_slab', { type: 'bottom', waterlogged: true }]]))
  assert.equal(digVerdict(logged, target, context(logged, { bot: at(0.5, 70, 5.5) })).reason, 'liquid_adjacent')
  const dry = await buildWorld(terrain(base, [[0, 61, 1, 'oak_slab', { type: 'bottom', waterlogged: false }]]))
  assert.equal(digVerdict(dry, target, context(dry, { bot: at(0.5, 70, 5.5) })).ok, true)
  const self = await buildWorld(terrain(base, [[0, 61, 0, 'oak_stairs', { facing: 'north', half: 'bottom', shape: 'straight', waterlogged: true }]]))
  assert.equal(digVerdict(self, target, context(self, { bot: at(0.5, 70, 5.5) })).reason, 'liquid_adjacent')
})

test('digVerdict refuses a chain of falling blocks above, hanging dripstone, ice, infested stone and spawners', async () => {
  const base = box(-3, 3, 55, 60, -3, 3, 'stone')
  const target = at(0, 58, 0)
  const verdict = async (extra, cell = target) => {
    const world = await buildWorld(terrain(base, extra))
    return digVerdict(world, cell, context(world, { bot: at(0.5, 70, 6.5) }))
  }
  const chain = await verdict([[0, 59, 0, 'sand'], [0, 60, 0, 'sand'], [0, 61, 0, 'gravel']])
  assert.deepEqual([chain.reason, chain.chain, chain.neighbour.name], ['gravity_above', 3, 'sand'])
  assert.equal((await verdict(box(0, 0, 59, 66, 0, 0, 'red_sand'))).chain, 6) // the scan stops at six blocks
  for (const name of ['white_concrete_powder', 'suspicious_gravel', 'anvil']) assert.equal((await verdict([[0, 59, 0, name]])).reason, 'gravity_above', name)
  // Sand beside the cell, or held up by a block that stays, is not above the cell.
  assert.equal((await verdict([[1, 58, 0, 'sand'], [0, 60, 0, 'sand']])).ok, true)
  assert.equal((await verdict([[0, 57, 0, 'pointed_dripstone', { thickness: 'tip', vertical_direction: 'down', waterlogged: false }]])).reason, 'dripstone_below')
  for (const name of ['ice', 'packed_ice', 'blue_ice', 'frosted_ice']) assert.equal((await verdict([[0, 58, 0, name]])).reason, 'melts_to_water', name)
  for (const name of ['infested_stone', 'infested_deepslate', 'infested_cobblestone']) assert.equal((await verdict([[0, 58, 0, name]])).reason, 'infested', name)
  // A spawner within 8 blocks of the cell refuses it; one 9 blocks away does not.
  const near = await buildWorld(terrain(base, box(-10, 10, 50, 50, -10, 10, 'stone'), [[8, 58, 0, 'spawner']]))
  const nearCtx = context(near, { bot: at(0.5, 70, 6.5) })
  assert.deepEqual(nearCtx.spawners.map(String), ['(8, 58, 0)'])
  assert.equal(digVerdict(near, target, nearCtx).reason, 'spawner_nearby')
  assert.equal(digVerdict(near, at(-1, 58, 0), nearCtx).ok, true)
  assert.throws(() => digVerdict(near, target, { botPosition: at(0.5, 70, 6.5), spawn }))
})

test('digVerdict refuses the bot\'s own columns above and below, the margin, its own table and a hostile in sight', async () => {
  const world = await buildWorld(terrain(box(-4, 4, 60, 63, -4, 4, 'grass_block'), box(-4, 4, 66, 66, -4, 4, 'stone'), [[3, 64, 3, 'crafting_table']]))
  const bot = at(0.5, 64, 0.5)
  const ctx = context(world, { bot, table: at(3, 64, 3) })
  assert.equal(digVerdict(world, at(0, 63, 0), ctx).reason, 'own_column') // its support
  assert.equal(digVerdict(world, at(0, 61, 0), ctx).reason, 'own_column') // deeper in its column
  assert.equal(digVerdict(world, at(0, 66, 0), ctx).reason, 'own_column') // the ceiling above its head
  assert.equal(digVerdict(world, at(1, 66, 0), ctx).ok, true)
  // Off centre, the corner column carries the hitbox too, above and below.
  const edge = { ...ctx, botPosition: at(0.8, 64, 0.5) }
  assert.equal(digVerdict(world, at(1, 63, 0), edge).reason, 'own_column')
  assert.equal(digVerdict(world, at(1, 66, 0), edge).reason, 'own_column')
  assert.equal(inHitboxColumns(at(0.5, 64, 0.5), at(1, 63, 0)), false)
  assert.equal(digVerdict(world, at(3, 64, 3), ctx).reason, 'own_table')
  assert.equal(digVerdict(world, at(3, 63, 3), ctx).reason, 'own_table')
  assert.equal(digVerdict(world, at(3, 62, 3), ctx).ok, true)
  // The cell centre must stay MAX_RADIUS - TARGET_MARGIN from spawn.
  const far = { ...ctx, spawn: { x: 0.5 - (MAX_RADIUS - TARGET_MARGIN) + 1.5, z: 0.5 } }
  assert.equal(digVerdict(world, at(2, 66, 0), far).reason, 'outside_margin') // centre 61.5 from spawn
  assert.equal(digVerdict(world, at(1, 66, 0), far).ok, true) // centre 60.5
  // A zombie 10 blocks away in the open refuses every dig; behind a wall, 13 blocks away, or a cow does not.
  const zombie = { name: 'zombie', position: at(10.5, 64, 0.5), height: 1.95 }
  const open = await buildWorld(terrain(box(-4, 14, 63, 63, -4, 4, 'grass_block'), [[2, 64, 2, 'stone']]))
  assert.equal(visibleHostiles(open, bot, [zombie]).length, 1)
  assert.equal(digVerdict(open, at(2, 64, 2), context(open, { entities: [zombie] })).reason, 'hostile_nearby')
  assert.equal(digVerdict(open, at(2, 64, 2), context(open, { entities: [{ ...zombie, position: at(13.5, 64, 0.5) }] })).ok, true)
  assert.equal(digVerdict(open, at(2, 64, 2), context(open, { entities: [{ ...zombie, name: 'cow' }] })).ok, true)
  const walled = await buildWorld(terrain(box(-4, 14, 63, 63, -4, 4, 'grass_block'), box(5, 5, 64, 68, -4, 4, 'stone'), [[2, 64, 2, 'stone']]))
  assert.equal(visibleHostiles(walled, bot, [zombie]).length, 0)
  assert.equal(digVerdict(walled, at(2, 64, 2), context(walled, { entities: { 7: zombie } })).ok, true)
  for (const name of ['zombie', 'skeleton', 'spider', 'creeper', 'silverfish', 'slime']) assert.equal(visibleHostiles(open, bot, [{ ...zombie, name }]).length, 1, name)
})

test('digVerdict checks in its documented order and adds the staircase classes', async () => {
  assert.deepEqual(DIG_REFUSALS.slice(0, 14), ['unloaded', 'nothing_to_dig', 'undiggable', 'own_table', 'own_column', 'outside_margin', 'liquid_adjacent',
    'gravity_above', 'dripstone_below', 'melts_to_water', 'infested', 'spawner_nearby', 'hostile_nearby', 'stone_source'])
  const names = ['oak_log', 'oak_leaves', 'stone', 'deepslate', 'coal_ore', 'iron_ore', 'chest', 'furnace', 'moss_block', 'dirt', 'granite', 'bedrock']
  const world = await buildWorld(terrain(box(-2, 14, 60, 63, -2, 2, 'stone'), names.map((name, i) => [i, 63, 0, name])))
  const ctx = { ...context(world, { bot: at(0.5, 70, -1.5) }), purpose: 'stair' }
  const reason = (i, extra = {}) => digVerdict(world, at(i, 63, 0), { ...ctx, ...extra }).reason
  assert.deepEqual(names.map((_, i) => reason(i)), ['log', 'leaves', 'stone_source', 'stone_source', 'ore', 'ore', 'container', 'container', 'not_dig_class', null, null, 'undiggable'])
  assert.equal(reason(0, { purpose: 'mine' }), null) // the class rules belong to the staircase only
  assert.equal(reason(5, { purpose: 'mine' }), 'wrong_tool')
  assert.equal(digVerdict(world, at(0, 70, 0), ctx).reason, 'nothing_to_dig')
  assert.equal(digVerdict(world, at(200, 63, 0), ctx).reason, 'unloaded')
  // A cell failing several checks reports the first in DIG_REFUSALS: own column before liquid and ice.
  const both = await buildWorld(terrain(box(-2, 2, 60, 63, -2, 2, 'stone'), [[0, 63, 0, 'ice'], [1, 63, 0, 'water']]))
  assert.equal(digVerdict(both, at(0, 63, 0), context(both)).reason, 'own_column')
  assert.equal(digVerdict(both, at(0, 63, 0), context(both, { bot: at(0.5, 70, -1.5) })).reason, 'liquid_adjacent')
  assert.throws(() => digVerdict(world, at(0, 63, 0), { ...ctx, purpose: 'build' }))
})

test('a drop rests on the first collision surface below and must land between feet - 1 and feet + 2', async () => {
  // The bot stands at feet 64 on grass; targets in the next column from feet - 2 to feet + 3.
  const world = await buildWorld(terrain(box(-2, 6, 55, 63, -2, 2, 'stone'), [[3, 64, 1, 'stone'],
    [3, 65, 1, 'stone'], [4, 65, 1, 'stone'], [4, 66, 1, 'stone'], [5, 67, 1, 'stone'], [5, 66, 1, 'oak_slab', { type: 'bottom', waterlogged: false }],
    [1, 67, -1, 'stone']]))
  const feet = 64
  const cases = [
    [at(1, 62, 0), 62], // feet - 2, in the ground: rests at feet - 2 (refused)
    [at(2, 63, 0), 63], // feet - 1
    [at(3, 64, 1), 64], // feet, resting on the block below it
    [at(3, 65, 1), 65], // feet + 1 on the block at feet
    [at(4, 66, 1), 66], // feet + 2 on the block at feet + 1
    [at(5, 67, 1), 66.5], // feet + 3 on a slab: rests at 66.5 (refused)
    [at(1, 67, -1), 64] // feet + 3 with open air below falls to the ground at feet (allowed)
  ]
  for (const [cell, rest] of cases) assert.equal(dropRestY(world, cell), rest, `${cell}`)
  assert.deepEqual(cases.map(([cell]) => inDropBand(dropRestY(world, cell), feet)), [false, true, true, true, true, false, true])
  const wet = await buildWorld([[0, 60, 0, 'stone'], [0, 61, 0, 'water']])
  assert.equal(dropRestY(wet, at(0, 62, 0)), null) // a fall through water is not followed
  assert.equal(dropRestY(wet, at(0, 200, 0)), null) // nothing within 16 blocks
  assert.equal(inDropBand(null, 64), false)
})

test('reach sees an exposed stone face and ignores buried stone, stone behind leaves, lookalikes and far blocks', async () => {
  const world = await buildWorld(terrain(box(-6, 8, 60, 62, -6, 6, 'stone'), box(-6, 8, 63, 63, -6, 6, 'grass_block'),
    [[2, 64, 0, 'stone'], [0, 64, 2, 'granite'], [-2, 64, -2, 'cobblestone'], [6, 64, 0, 'stone'], [-3, 64, 2, 'infested_stone']],
    // A stone block wrapped in leaves on every side facing the bot; its open east face looks away.
    [[3, 64, -3, 'stone'], [2, 64, -3, 'oak_leaves'], [3, 65, -3, 'oak_leaves'], [3, 64, -2, 'oak_leaves'], [3, 64, -4, 'oak_leaves']]))
  const ctx = context(world)
  const found = reachableStone(world, ctx)
  assert.deepEqual(found.map(c => `${c.position}`), ['(2, 64, 0)', '(-2, 64, -2)'])
  assert.deepEqual(found.map(c => c.drop), ['cobblestone', 'cobblestone'])
  assert(found[0].distance < found[1].distance)
  assert.equal(found[0].rest_y, 64)
  assert.deepEqual(found[0].faces.sort(), ['east', 'north', 'south', 'top', 'west']) // all but the grass below
  // Buried stone has no open face; the support is in the bot's own column.
  assert.deepEqual(openFaces(world, world.getBlock(at(1, 62, 0))), [])
  assert.equal(found.some(c => c.position.y < 64), false)
  // The same rule run by the observation and by MINE_STONE: one function, the same answer.
  assert.deepEqual(reachableStone(world, ctx).map(c => `${c.position}`), found.map(c => `${c.position}`))
  // Without the leaves the wrapped block is visible and in reach.
  const bare = await buildWorld(terrain(box(-6, 8, 60, 62, -6, 6, 'stone'), box(-6, 8, 63, 63, -6, 6, 'grass_block'), [[3, 64, -3, 'stone']]))
  assert.deepEqual(reachableStone(bare, context(bare)).map(c => `${c.position}`), ['(3, 64, -3)'])
})

test('the ray fan finds exposed stone in sight and ignores buried stone, stone behind leaves and lookalikes', async (t) => {
  assert.equal(FAN_ELEVATIONS.length, 13)
  assert.deepEqual([FAN_ELEVATIONS[0], FAN_ELEVATIONS[12]], [-60, 60])
  const blocks = terrain(box(-40, 40, 55, 62, -40, 40, 'stone'), box(-40, 40, 63, 63, -40, 40, 'grass_block'),
    box(10, 12, 64, 68, -4, 4, 'stone'), // a cliff 9.5 blocks east
    box(-12, -10, 64, 66, -2, 2, 'granite'), // a lookalike cliff to the west
    box(-2, 2, 64, 66, -14, -12, 'stone'), box(-3, 3, 64, 67, -10, -10, 'oak_leaves'), // stone behind a leaf wall to the north
    box(-2, 2, 64, 66, 36, 38, 'stone') // beyond the 32-block rays
  )
  const world = await buildWorld(blocks)
  const ctx = context(world)
  const { targets, stats } = visibleStoneTargets(world, ctx)
  assert(targets.length > 0)
  assert(targets.every(target => target.position.x >= 10 && target.position.x <= 12 && target.block.name === 'stone'), targets.map(t => `${t.position}`).join(' '))
  assert(targets.every(target => target.position.y >= 64)) // the stone under the grass is never a first hit
  assert.equal(targets[0].face, 'west')
  for (let i = 1; i < targets.length; i++) assert(targets[i].distance >= targets[i - 1].distance)
  assert.equal(stats.rays, 32 * 13)
  t.diagnostic(`ray fan: ${stats.rays} rays, ${stats.reads} block reads, ${stats.hits} distinct first hits, ${targets.length} targets, ${stats.ms.toFixed(1)} ms`)
  // A failed target column is skipped until the memory releases it.
  const memory = new StoneMemory()
  memory.next(targets)
  memory.failed(targets[0].block)
  const after = visibleStoneTargets(world, ctx, { memory }).targets
  assert(after.length > 0 && after.every(target => `${target.position.x},${target.position.z}` !== `${targets[0].position.x},${targets[0].position.z}`))
  // Without the leaf wall the northern stone is seen.
  const open = await buildWorld(blocks.filter(block => block[3] !== 'oak_leaves'))
  assert(visibleStoneTargets(open, context(open)).targets.some(target => target.position.z <= -12))
  // A hostile in sight empties the list, because every target must pass digVerdict.
  const hostile = context(world, { entities: [{ name: 'zombie', position: at(5.5, 64, 0.5), height: 1.95 }] })
  assert.equal(visibleStoneTargets(world, hostile).targets.length, 0)
  // The fan reports the first hit of each ray only, with its distance.
  const fan = rayFan(world, at(0.5, 65.62, 0.5), { headings: 4, elevations: [0] })
  assert.deepEqual(fan.hits.map(hit => hit.block.name).sort(), ['granite', 'oak_leaves', 'stone'])
})

test('stone memory blocks failed columns like the log search memory and can be queried without advancing', () => {
  const memory = new StoneMemory()
  assert(memory instanceof SearchMemory)
  const first = { position: { x: 1, y: 60, z: 1 } }
  const second = { position: { x: 2, y: 61, z: 2 } }
  assert.equal(memory.peek([first, second]), first)
  assert.equal(memory.attempt, 0)
  assert.equal(memory.next([first, second]), first)
  memory.failed(first)
  assert.equal(memory.blocked(first), true)
  assert.equal(memory.blocked({ position: { x: 1, y: 62, z: 1 } }), true) // same column
  assert.equal(memory.peek([first, second]), second)
  assert.equal(memory.next([first, second]), second)
  memory.failed(second)
  assert.equal(memory.peek([first, second]), null) // two failures in a row force exploration
  assert.equal(memory.next([first, second]), null)
  for (let i = 0; i < 5; i++) assert.equal(memory.next([first, second]), null)
  assert.equal(memory.peek([first, second]), first)
  assert.equal(memory.next([first, second]), first)
})

test('the staircase on flat ground digs 1, 2, 3, 3 and 2 cells and exposes stone level with the floor', async () => {
  const world = await buildWorld(workedExample())
  const ctx = stairContext(world)
  const stair = planStaircase(world, at(0, 64, 0), '+x', ctx, { stopWhen: reachStop(ctx) })
  assert.deepEqual(stair.dig_counts, [1, 2, 3, 3, 2])
  assert.equal(stair.stopped, 'stone_exposed')
  assert.deepEqual(stair.steps.slice(0, 4).map(step => `${step.landing}`), ['(1, 63, 0)', '(2, 62, 0)', '(3, 61, 0)', '(4, 60, 0)'])
  assert.deepEqual(stair.steps.slice(0, 4).map(step => step.floor.name), ['dirt', 'dirt', 'dirt', 'stone'])
  const last = stair.steps[4]
  assert.deepEqual([last.move, `${last.exposes}`, last.exposed_block], [false, '(5, 59, 0)', 'stone'])
  // Every dig is by hand: 11 digs (3 grass at 0.9 s, 8 dirt at 0.75 s), 8.7 s in all, no step over 8 s.
  const digs = stair.steps.flatMap(step => step.digs)
  assert.equal(digs.length, 11)
  assert(digs.every(dig => dig.tool === 'hand' && ['grass_block', 'dirt'].includes(dig.name)))
  assert.equal(digs.reduce((total, dig) => total + dig.ms, 0), 8700)
  assert(stair.steps.every(step => step.dig_ms <= 8000))
  // The exposed block is the first MINE_STONE target, its drop one step down at feet - 1.
  const reach = reachableStone(stair.view, { ...ctx, botPosition: at(4.5, 60, 0.5) })
  assert.deepEqual(reach.map(c => [`${c.position}`, c.rest_y, c.faces.join()]), [['(5, 59, 0)', 59, 'top']])
  // The heading is chosen among the four by rotation when the ground is flat.
  assert.equal(chooseStairHeading(world, at(0, 64, 0), ctx).heading, '+x')
  assert.equal(chooseStairHeading(world, at(0, 64, 0), ctx, 1).heading, '+z')
  assert.equal(chooseStairHeading(world, at(0, 64, 0), ctx, 3).heading, '-z')
})

test('the staircase prefers a slope and refuses sand ceilings, caves, rivers, the radius edge, the depth limit and unloaded columns', async () => {
  const base = workedExample()
  // A slope rising to +z: two solid cells at H and M.
  const slope = await buildWorld(terrain(base, box(-6, 12, 64, 64, 1, 6, 'dirt'), box(-6, 12, 65, 65, 1, 6, 'grass_block')))
  const sctx = stairContext(slope)
  const choice = chooseStairHeading(slope, at(0, 64, 0), sctx)
  assert.deepEqual([choice.heading, choice.plan.solid, choice.plan.digs.length], ['+z', 2, 3])
  // A sand ceiling over step 4.
  const sand = await buildWorld(terrain(base, [[4, 62, 0, 'sand'], [4, 63, 0, 'sand']]))
  const sandStair = planStaircase(sand, at(0, 64, 0), '+x', stairContext(sand))
  assert.deepEqual([sandStair.stopped, sandStair.dig_counts, sandStair.steps[3].block, sandStair.steps[3].neighbour], ['gravity_above', [1, 2, 3], 'sand', 'sand'])
  // A cave under step 3's landing.
  const cave = await buildWorld(terrain(base, box(2, 4, 58, 60, -1, 1, 'cave_air')))
  const caveStair = planStaircase(cave, at(0, 64, 0), '+x', stairContext(cave))
  assert.deepEqual([caveStair.stopped, caveStair.dig_counts, caveStair.steps[2].floor], ['cave_below', [1, 2], 'cave_air'])
  // A river along z = 1: every heading touching it is refused, the one away from it is chosen.
  const river = await buildWorld(terrain(base, box(-6, 12, 62, 63, 1, 1, 'water')))
  const riverChoice = chooseStairHeading(river, at(0, 64, 0), stairContext(river))
  assert.deepEqual([riverChoice.heading, riverChoice.refused], ['-z', { '+x': 'liquid_adjacent', '+z': 'liquid_adjacent', '-x': 'liquid_adjacent' }])
  // The radius edge: from 58 blocks out, column 4 has its centre 62 blocks from spawn.
  const edgeSpawn = { x: -57.5, z: 0.5 }
  const flat = await buildWorld(base)
  const edge = planStaircase(flat, at(0, 64, 0), '+x', stairContext(flat, { home: edgeSpawn }))
  assert.deepEqual([edge.stopped, edge.dig_counts], ['outside_margin', [1, 2, 3]])
  // The depth limit: 12 below the stage start.
  const deep = await buildWorld(terrain(box(-6, 12, 40, 63, -6, 6, 'dirt')))
  const deepStair = planStaircase(deep, at(0, 64, 0), '+x', { ...stairContext(deep), startY: 64 + MAX_STAIR_DEPTH - 2 })
  assert.deepEqual([deepStair.stopped, deepStair.dig_counts, `${deepStair.feet}`], ['depth_limit', [1, 2], '(2, 62, 0)'])
  // An unloaded column on the far side of a chunk edge.
  const edgeWorld = await buildWorld(workedExample(8, 15, -4, 4))
  edgeWorld.unloadColumn(1, 0)
  assert.equal(planStairStep(edgeWorld, at(15, 64, 0), '+x', { ...stairContext(edgeWorld, { bot: at(15.5, 64, 0.5), home: { x: 15.5, z: 0.5 } }) }).reason, 'unloaded')
  // Headings within one column of the bot's own table are dropped.
  const table = await buildWorld(terrain(base, [[2, 64, 0, 'crafting_table']]))
  const tableChoice = chooseStairHeading(table, at(0, 64, 0), stairContext(table, { table: at(2, 64, 0) }))
  assert.deepEqual([tableChoice.heading, tableChoice.refused], ['+z', { '+x': 'near_own_table' }])
  // A log, an ore or a stone source in the way refuses the heading; stone under one dug cell is exposed.
  for (const [name, reason] of [['oak_log', 'log'], ['coal_ore', 'ore'], ['oak_leaves', 'leaves']]) {
    const world = await buildWorld(terrain(base, [[1, 63, 0, name]]))
    assert.equal(planStairStep(world, at(0, 64, 0), '+x', stairContext(world)).reason, reason, name)
  }
  const ahead = await buildWorld(terrain(base, [[1, 64, 0, 'stone']]))
  assert.equal(planStairStep(ahead, at(0, 64, 0), '+x', stairContext(ahead)).reason, 'stone_source')
  const knob = await buildWorld(terrain(base, [[1, 64, 0, 'stone'], [1, 65, 0, 'dirt']]))
  const exposing = planStairStep(knob, at(0, 64, 0), '+x', stairContext(knob))
  assert.deepEqual([exposing.ok, exposing.move, exposing.digs.map(dig => dig.name), `${exposing.exposes}`], [true, false, ['dirt'], '(1, 64, 0)'])
  // Granite is dug with the pickaxe.
  const granite = await buildWorld(terrain(base, [[1, 63, 0, 'granite']]))
  assert.deepEqual(planStairStep(granite, at(0, 64, 0), '+x', stairContext(granite)).digs.map(dig => [dig.name, dig.tool, dig.ms]), [['granite', 'pickaxe', 1150]])
  for (const bad of [[at(0.5, 64, 0), '+x'], [at(0, 64, 0), 'up']]) assert.throws(() => planStairStep(granite, bad[0], bad[1], stairContext(granite)))
  assert.throws(() => planStairStep(granite, at(0, 64, 0), '+x', { ...stairContext(granite), startY: undefined }))
})

test('every staircase step has a drop-down move and a move back up under the real pathfinder rules', async () => {
  const world = await buildWorld(workedExample())
  const ctx = stairContext(world)
  const stair = planStaircase(world, at(0, 64, 0), '+x', ctx)
  const feet = [at(0, 64, 0), ...stair.steps.filter(step => step.move).map(step => step.landing)]
  assert.equal(feet.length, 5)
  // Checked again on the finished staircase: later digs keep earlier moves open.
  for (let i = 1; i < feet.length; i++) assert.deepEqual(reversible(stair.view, feet[i - 1], feet[i]), { down: true, up: true }, `${feet[i - 1]} -> ${feet[i]}`)
  const settings = strictMovements({ registry, entity: { position: at(0.5, 64, 0.5) } }, Movements, spawn)
  for (const flag of ['canDig', 'allow1by1towers', 'allowParkour', 'allowSprinting', 'allowFreeMotion']) assert.equal(settings[flag], false)
  assert.equal(settings.maxDropDown, 2)
  assert(settings.blocksToAvoid.has(registry.blocksByName.pointed_dripstone.id))
  // Step 3 with M and L dug but not H (underground there): no drop in and no climb back out.
  const two = planStaircase(world, at(0, 64, 0), '+x', ctx, { maxSteps: 2 })
  assert.deepEqual(reversible(clearedView(two.view, [at(3, 62, 0), at(3, 61, 0)]), at(2, 62, 0), at(3, 61, 0)), { down: false, up: false })
  assert.deepEqual(reversible(clearedView(two.view, [at(3, 63, 0), at(3, 62, 0), at(3, 61, 0)]), at(2, 62, 0), at(3, 61, 0)), { down: true, up: true })
  const noTop = await buildWorld(terrain(workedExample(), [[1, 65, 0, 'dirt']]))
  assert.deepEqual(reversible(clearedView(noTop, [at(1, 64, 0), at(1, 63, 0)]), at(0, 64, 0), at(1, 63, 0)), { down: false, up: false })
  // L left in place: there is no cell to drop into.
  assert.equal(reversible(clearedView(world, [at(1, 64, 0)]), at(0, 64, 0), at(1, 63, 0)).down, false)
  // A two-block feet drop (floor three below the old feet) is beyond maxDropDown 2.
  assert.deepEqual(reversible(clearedView(world, [at(1, 63, 0), at(1, 62, 0)]), at(0, 64, 0), at(1, 62, 0)), { down: false, up: false })
  const refusing = planStairStep(world, at(0, 64, 0), '+x', { ...ctx, reversible: () => ({ down: true, up: false }) })
  assert.equal(refusing.reason, 'not_reversible')
})

test('a stone pickaxe is filled from 3 cobblestone or from mixed materials and refused short of materials or sticks', () => {
  const Recipe = fromMineflayer('prismarine-recipe')(registry).Recipe
  const templates = Recipe.find(registry.itemsByName.stone_pickaxe.id)
  assert.equal(templates.length, 3)
  const id = name => registry.itemsByName[name].id
  const template = templates.find(recipe => recipe.inShape.flat().some(item => item.id === id('cobblestone')))
  const materialRow = recipe => recipe.inShape.find(row => row.every(item => item.id !== -1 && item.id !== id('stick'))).map(item => registry.items[item.id].name)
  const fill = items => fillStoneRecipe(template, items.map(([name, count]) => ({ type: id(name), count })))
  assert.deepEqual(materialRow(fill([['cobblestone', 3], ['stick', 2]])), ['cobblestone', 'cobblestone', 'cobblestone'])
  const mixed = fill([['cobblestone', 2], ['cobbled_deepslate', 1], ['stick', 2]])
  assert.deepEqual(materialRow(mixed), ['cobblestone', 'cobblestone', 'cobbled_deepslate'])
  assert.equal(mixed.inShape.flat().filter(item => item.id === id('stick')).length, 2)
  assert.equal(mixed.requiresTable, true)
  // Any template works, whatever material it lists; the template itself is never changed.
  const blackstoneTemplate = templates.find(recipe => recipe.inShape.flat().some(item => item.id === id('blackstone')))
  assert.deepEqual(materialRow(fillStoneRecipe(blackstoneTemplate, [{ type: id('cobbled_deepslate'), count: 1 }, { type: id('blackstone'), count: 2 }, { type: id('stick'), count: 5 }])),
    ['cobbled_deepslate', 'blackstone', 'blackstone'])
  assert.deepEqual(materialRow(blackstoneTemplate), ['blackstone', 'blackstone', 'blackstone'])
  assert.equal(fill([['cobblestone', 2], ['stick', 2]]), null)
  assert.equal(fill([['cobblestone', 1], ['cobbled_deepslate', 1], ['stick', 4]]), null)
  assert.equal(fill([['cobblestone', 3], ['stick', 1]]), null)
  assert.equal(fill([['cobblestone', 3], ['stone', 3], ['granite', 3], ['stick', 1]]), null)
  assert.equal(fillStoneRecipe(null, []), null)
})

test('Mineflayer 4.39 keeps 1.21.4 sky light from map_chunk and update_light, and lit faces can be required', async () => {
  // A real map_chunk packet through the pinned 1.21.4 codec into Mineflayer's own blocks plugin.
  const version = '1.21.4'
  const Chunk = fromMineflayer('prismarine-chunk')(version)
  const protocol = fromMineflayer('minecraft-protocol')
  const source = new Chunk({ minY: -64, worldHeight: 384 })
  for (let x = 0; x < 16; x++) for (let z = 0; z < 16; z++) for (let y = 50; y < 64; y++) source.setBlockStateId(at(x, y, z), registry.blocksByName.stone.defaultState)
  source.setBlockStateId(at(5, 55, 5), 0)
  for (let x = 0; x < 16; x++) for (let z = 0; z < 16; z++) for (let y = 64; y < 80; y++) source.setSkyLight(at(x, y, z), 15)
  source.setSkyLight(at(3, 70, 3), 9)
  const serializer = protocol.createSerializer({ state: 'play', isServer: true, version })
  const deserializer = protocol.createDeserializer({ state: 'play', isServer: false, version })
  const roundTrip = (name, params) => deserializer.parsePacketBuffer(serializer.createPacketBuffer({ name, params })).data.params
  const bot = new EventEmitter()
  bot._client = new EventEmitter()
  Object.assign(bot, { registry, version, supportFeature: registry.supportFeature, game: { minY: -64, height: 384, dimension: 'overworld' } })
  require(require.resolve('mineflayer/lib/plugins/blocks', { paths: [mineflayerEntry] }))(bot, { version, hideErrors: true })
  bot._client.emit('login', { worldState: { dimension: 0, name: 'minecraft:overworld' } })
  bot._client.emit('map_chunk', roundTrip('map_chunk', { x: 0, z: 0, heightmaps: { type: 'compound', name: '', value: {} }, chunkData: source.dump(), blockEntities: [], ...source.dumpLight() }))
  assert.deepEqual([[2, 70, 2], [3, 70, 3], [5, 64, 5], [5, 55, 5]].map(p => bot.world.getSkyLight(at(...p))), [15, 9, 15, 0])
  assert.equal(bot.blockAt(at(2, 70, 2)).skyLight, 15)
  assert.equal(bot.blockAt(at(5, 55, 5)).name, 'air')
  const update = new Chunk({ minY: -64, worldHeight: 384 })
  update.setSkyLight(at(5, 55, 5), 4)
  bot._client.emit('update_light', roundTrip('update_light', { chunkX: 0, chunkZ: 0, ...update.dumpLight() }))
  assert.deepEqual([bot.world.getSkyLight(at(5, 55, 5)), bot.world.getSkyLight(at(2, 70, 2))], [4, 15])

  // A dark tunnel: stone walls in reach count only once their open cells are lit.
  const tunnel = await buildWorld(terrain(box(-4, 8, 56, 64, -4, 4, 'stone'), box(0, 4, 60, 61, 0, 0, 'air')))
  const ctx = context(tunnel, { bot: at(0.5, 60, 0.5), startY: 60 })
  assert(reachableStone(tunnel, ctx).length > 0)
  assert.equal(reachableStone(tunnel, ctx, { litOnly: true }).length, 0)
  assert.equal(visibleStoneTargets(tunnel, ctx, { litOnly: true }).targets.length, 0)
  assert(visibleStoneTargets(tunnel, ctx).targets.length > 0)
  for (let x = 0; x <= 4; x++) for (const y of [60, 61]) tunnel.setSkyLight(at(x, y, 0), 7)
  assert.deepEqual(reachableStone(tunnel, ctx, { litOnly: true }).map(c => `${c.position}`), reachableStone(tunnel, ctx).map(c => `${c.position}`))
  assert(visibleStoneTargets(tunnel, ctx, { litOnly: true }).targets.length > 0)
  // Block light counts as lit too.
  const torch = await buildWorld(terrain(box(-4, 8, 56, 64, -4, 4, 'stone'), box(0, 4, 60, 61, 0, 0, 'air')))
  torch.setBlockLight(at(1, 60, 0), 12)
  assert(reachableStone(torch, context(torch, { bot: at(0.5, 60, 0.5), startY: 60 }), { litOnly: true }).length > 0)
})

test('spawners are found through chunk palettes, as Mineflayer\'s findBlocks does', async () => {
  const world = await buildWorld(terrain(box(-20, 20, 40, 41, -20, 20, 'stone'), [[30, 50, 0, 'spawner'], [5, 41, 5, 'spawner'], [-45, 60, 0, 'trial_spawner']]))
  const found = findBlocksByName(world, at(0.5, 64, 0.5), 40, SPAWNER_BLOCKS).map(String).sort()
  assert.deepEqual(found, ['(30, 50, 0)', '(5, 41, 5)']) // the trial spawner is 45 blocks away
  assert.deepEqual(findBlocksByName(world, at(-20, 60, 0), 30, SPAWNER_BLOCKS).map(String), ['(-45, 60, 0)'])
  assert.deepEqual(findBlocksByName(world, at(0.5, 64, 0.5), 10, SPAWNER_BLOCKS), [])
})

test('stone-helpers requires the bridge at load time, the bridge never requires it at top level, and neither starts a bot', () => {
  const dir = __dirname
  for (const order of [['./stone-helpers', './resource-bridge'], ['./resource-bridge', './stone-helpers']]) {
    const script = `const a = require(${JSON.stringify(order[0])}); const b = require(${JSON.stringify(order[1])}); ` +
      'const s = require("./stone-helpers"); const r = require("./resource-bridge"); ' +
      'if (typeof s.digVerdict !== "function" || typeof r.canSeeFace !== "function") process.exit(3); console.log("loaded")'
    const run = spawnSync(process.execPath, ['-e', script], { cwd: dir, env: process.env, encoding: 'utf8', timeout: 20000 })
    assert.equal(run.status, 0, `${order.join(' then ')}: ${run.stderr}`)
    assert.equal(run.stdout.trim(), 'loaded')
  }
  // Requiring the bridge alone does not load the stone helpers.
  const probe = 'require("./resource-bridge"); console.log(Object.keys(require.cache).some(f => f.endsWith("stone-helpers.js")))'
  const alone = spawnSync(process.execPath, ['-e', probe], { cwd: dir, env: process.env, encoding: 'utf8', timeout: 20000 })
  assert.equal(alone.stdout.trim(), 'false')
  const source = fs.readFileSync(path.join(dir, 'resource-bridge.js'), 'utf8')
  assert.doesNotMatch(source, /^(const|let|var)[^\n]*require\(['"]\.\/stone-helpers['"]\)/m)
  const helpers = fs.readFileSync(path.join(dir, 'stone-helpers.js'), 'utf8')
  assert.doesNotMatch(helpers, /\.dig\s*\(|\.chat\s*\(|\.craft\s*\(|\.equip\s*\(|setControlState|pathfinder\.goto|placeBlock/)
})
