'use strict'

// Pure helpers for the stone-tools stage: block classes, tool choice, the one
// dig safety check, stone perception and the staircase planner. Nothing here
// moves the bot, digs, crafts or sends a packet; every function reads a
// prismarine world (the offline test world or bot.world, both WorldSync).
const { Vec3 } = require('vec3')
const {
  SearchMemory, canSeeFace, passableCell, walkableFloor, collisionTop, withinRadius, eyesAt, safeMovements, isLeaves,
  interactiveSupport, HITBOX_HALF_WIDTH, MAX_RADIUS, TARGET_MARGIN
} = require('./resource-bridge')

// The bridge is pinned to 1.21.4 (parseArgs), so the data come from the same
// pinned packages Mineflayer itself loads.
const mineflayerEntry = require.resolve('mineflayer')
const fromMineflayer = name => require(require.resolve(name, { paths: [mineflayerEntry] }))
const registry = fromMineflayer('prismarine-registry')('1.21.4')
const Block = fromMineflayer('prismarine-block')(registry)
const { RaycastIterator } = fromMineflayer('prismarine-world').iterators

// Overworld blocks searched for, and the recipe material each gives with a wooden pickaxe.
const STONE_SOURCES = ['stone', 'cobblestone', 'deepslate', 'cobbled_deepslate']
const STONE_DROPS = { stone: 'cobblestone', cobblestone: 'cobblestone', deepslate: 'cobbled_deepslate', cobbled_deepslate: 'cobbled_deepslate', blackstone: 'blackstone' }
// Inventory items that fill the top row of a stone pickaxe recipe.
const STONE_MATERIALS = ['cobblestone', 'cobbled_deepslate', 'blackstone']
// Pickaxe blocks that drop themselves; never targets.
const STONE_LOOKALIKES = ['granite', 'andesite', 'diorite', 'tuff', 'calcite', 'sandstone', 'terracotta', 'mossy_cobblestone']
// What DIG_DOWN may dig, by hand (a slot with no tool) or with the pickaxe.
const HAND_DIG_BLOCKS = ['grass_block', 'dirt', 'coarse_dirt', 'rooted_dirt', 'podzol', 'mycelium', 'clay', 'mud', 'sand', 'red_sand', 'gravel', 'snow', 'snow_block']
const PICKAXE_DIG_BLOCKS = ['granite', 'diorite', 'andesite', 'tuff', 'calcite', 'sandstone', 'red_sandstone', 'terracotta']
const GRAVITY_BLOCKS = ['sand', 'red_sand', 'gravel', 'suspicious_sand', 'suspicious_gravel', 'anvil', 'chipped_anvil', 'damaged_anvil']
// Liquid cells; any block with waterlogged=true counts too.
const LIQUID_BLOCKS = ['water', 'lava', 'bubble_column', 'seagrass', 'tall_seagrass', 'kelp', 'kelp_plant']
const ICE_BLOCKS = ['ice', 'packed_ice', 'blue_ice', 'frosted_ice']
const SPAWNER_BLOCKS = ['spawner', 'trial_spawner']
const AIR_BLOCKS = ['air', 'cave_air', 'void_air']
const STONE_PICKAXE_REQUIREMENT = { materials: 3, sticks: 2, table: true }
const WOODEN_PICKAXE = registry.itemsByName.wooden_pickaxe.id

const GRAVITY_SCAN = 6
const SPAWNER_RADIUS = 8
const HOSTILE_RADIUS = 12
const HOSTILE_STOP_RADIUS = 6
const MAX_DIG_MS = 4000
const MAX_STEP_DIG_MS = 8000
const MAX_STAIR_DEPTH = 12
const MAX_STAIR_STEPS = 6
const REACH_SCAN_RADIUS = 6
const REACH_DISTANCE = 4.5
const FAN_HEADINGS = 32
const FAN_ELEVATIONS = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
const FAN_RANGE = 32
const STAIR_HEADINGS = { '+x': [1, 0], '+z': [0, 1], '-x': [-1, 0], '-z': [0, -1] }
// Face names as visibilitySamples uses them, and prismarine-world's BlockFace numbers.
const FACE_OFFSETS = { west: [-1, 0, 0], east: [1, 0, 0], bottom: [0, -1, 0], top: [0, 1, 0], north: [0, 0, -1], south: [0, 0, 1] }
const RAY_FACES = ['bottom', 'top', 'north', 'south', 'west', 'east']
// digVerdict checks in this order and reports the first that fails.
const DIG_REFUSALS = ['unloaded', 'nothing_to_dig', 'undiggable', 'own_table', 'own_column', 'outside_margin', 'liquid_adjacent', 'gravity_above',
  'dripstone_below', 'melts_to_water', 'infested', 'spawner_nearby', 'hostile_nearby', 'stone_source', 'log', 'leaves', 'ore', 'container',
  'not_dig_class', 'wrong_tool']
const HOSTILE_NAMES = new Set(registry.entitiesArray.filter(entity => entity.type === 'hostile' || entity.category === 'Hostile mobs').map(entity => entity.name))
const AIR_STATE = registry.blocksByName.air.defaultState
const HITBOX_CORNERS = [[0, 0], [HITBOX_HALF_WIDTH, HITBOX_HALF_WIDTH], [HITBOX_HALF_WIDTH, -HITBOX_HALF_WIDTH], [-HITBOX_HALF_WIDTH, HITBOX_HALF_WIDTH], [-HITBOX_HALF_WIDTH, -HITBOX_HALF_WIDTH]]

function isStoneSource (name) { return STONE_SOURCES.includes(name) }
function isStoneMaterial (name) { return STONE_MATERIALS.includes(name) }
function stoneDrop (name) { return STONE_DROPS[name] || null }
function isInfested (name) { return typeof name === 'string' && name.startsWith('infested_') }
function isGravityBlock (name) { return GRAVITY_BLOCKS.includes(name) || (typeof name === 'string' && name.endsWith('_concrete_powder')) }
function isIce (name) { return ICE_BLOCKS.includes(name) }
function isOre (name) { return typeof name === 'string' && (name.endsWith('_ore') || name === 'ancient_debris') }
function isLogLike (name) { return typeof name === 'string' && /_(log|wood|stem|hyphae)$/.test(name) }
function digClass (name) {
  if (HAND_DIG_BLOCKS.includes(name)) return 'hand'
  if (PICKAXE_DIG_BLOCKS.includes(name)) return 'pickaxe'
  return null
}
function isLiquidBlock (block) {
  if (!block) return false
  if (LIQUID_BLOCKS.includes(block.name)) return true
  try { return block.getProperties?.().waterlogged === true } catch (_) { return false }
}
// An empty cell a player could look through at a face: no collision box and no liquid.
function openCell (block) { return Boolean(block) && block.boundingBox === 'empty' && !isLiquidBlock(block) }
function isHostile (name) { return HOSTILE_NAMES.has(name) }

function stoneInventoryCounts (inventory) {
  const counts = { stone_materials: 0, stone_pickaxes: 0 }
  for (const [name, amount] of Object.entries(inventory)) {
    if (!Number.isInteger(amount) || amount < 0) throw new Error('Invalid inventory quantity')
    if (isStoneMaterial(name)) counts.stone_materials += amount
    if (name === 'stone_pickaxe') counts.stone_pickaxes += amount
  }
  return counts
}

// The pickaxe for pickaxe blocks, a slot with no tool for everything else
// (no durability cost). Dig times use prismarine-block's vanilla formula for a
// bot standing on the ground, out of water, with no enchantments or effects.
function pickaxeBlock (block) {
  const material = block?.material || ''
  return material.includes('mineable/pickaxe') || material.includes('incorrect_for_')
}
function digTimeMs (block, itemType, { onGround = true } = {}) {
  return block.digTime(itemType ?? null, false, false, !onGround, [], {})
}
function chooseTool (block, { pickaxeType = WOODEN_PICKAXE } = {}) {
  const pickaxe = pickaxeBlock(block)
  const type = pickaxe ? pickaxeType : null
  // Only the pickaxe has to harvest: a hand-dug block (snow) needs no drop.
  const harvestable = !pickaxe || !block.harvestTools || Boolean(block.harvestTools[pickaxeType])
  return { tool: pickaxe ? 'pickaxe' : 'hand', type, ms: digTimeMs(block, type), harvestable }
}

// The columns that carry any part of the bot's 0.6-wide hitbox, at any height:
// underHitbox extended upwards, because a block dug above the head can let
// gravel fall onto the bot.
function inHitboxColumns (botPosition, cell) {
  return HITBOX_CORNERS.some(([dx, dz]) => Math.floor(botPosition.x + dx) === cell.x && Math.floor(botPosition.z + dz) === cell.z)
}

function cellKey (position) { return `${position.x},${position.y},${position.z}` }
function centreOf (feet) { return new Vec3(feet.x + 0.5, feet.y, feet.z + 0.5) }
function neighbour (position, face) {
  const [dx, dy, dz] = FACE_OFFSETS[face]
  return new Vec3(position.x + dx, position.y + dy, position.z + dz)
}
function lightAt (world, position) {
  const sky = typeof world.getSkyLight === 'function' ? world.getSkyLight(position) : 0
  const block = typeof world.getBlockLight === 'function' ? world.getBlockLight(position) : 0
  return Math.max(sky || 0, block || 0)
}

// Faces of a block next to an open cell, optionally only those whose open
// cell has sky or block light above 0 (decision 9).
function openFaces (world, block, { litOnly = false } = {}) {
  return Object.keys(FACE_OFFSETS).filter(face => {
    const position = neighbour(block.position, face)
    return openCell(world.getBlock(position)) && (!litOnly || lightAt(world, position) > 0)
  })
}

// The nearest collision surface of `block` hit by the ray, or null. Shapes
// behind the origin are ignored, as in canSeeFace.
function shapeHit (block, iterator, origin, direction, range) {
  let best = null
  for (const shape of block.shapes || []) {
    const intersection = iterator.intersect([shape], block.position)
    if (!intersection) continue
    const t = intersection.pos.minus(origin).dot(direction)
    if (t >= -1e-9 && t <= range && (!best || t < best.t)) best = { t, face: intersection.face }
  }
  return best
}

function clearLine (world, from, to) {
  const delta = to.minus(from)
  const range = delta.norm()
  if (range < 1e-3) return true
  const direction = delta.scaled(1 / range)
  return !world.raycast(from, direction, range, (candidate, iterator) => shapeHit(candidate, iterator, from, direction, range) !== null)
}

// Hostile mobs within `radius` of the bot that have a clear line from the
// bot's eyes to their middle or their head. digVerdict reads this list as
// ctx.hostiles, so it is computed once per scan, not per cell.
function visibleHostiles (world, botPosition, entities, radius = HOSTILE_RADIUS) {
  const eyes = eyesAt(botPosition)
  return Object.values(entities || {}).filter(entity => entity?.position && isHostile(entity.name) && entity.position.distanceTo(botPosition) <= radius &&
    [0.5, 0.9].some(fraction => clearLine(world, eyes, entity.position.offset(0, (Number.isFinite(entity.height) ? entity.height : 1.8) * fraction, 0))))
}

// Positions of blocks with one of `names` within `radius` of `centre`. Chunk
// sections whose palette holds none of the names are skipped, as Mineflayer's
// findBlocks does, so a 40-block spawner scan stays cheap on bot.world.
function findBlocksByName (world, centre, radius, names) {
  const wanted = new Set(names)
  const states = new Set(registry.blocksArray.filter(block => wanted.has(block.name)).flatMap(block => {
    const ids = []
    for (let id = block.minStateId; id <= block.maxStateId; id++) ids.push(id)
    return ids
  }))
  const c = new Vec3(Math.floor(centre.x), Math.floor(centre.y), Math.floor(centre.z))
  const found = []
  for (let cx = Math.floor((c.x - radius) / 16); cx <= Math.floor((c.x + radius) / 16); cx++) {
    for (let cz = Math.floor((c.z - radius) / 16); cz <= Math.floor((c.z + radius) / 16); cz++) {
      const column = typeof world.getColumn === 'function' ? world.getColumn(cx, cz) : null
      if (!column) continue
      const minY = Number.isFinite(column.minY) ? column.minY : -64
      for (let sy = Math.max(0, (c.y - radius - minY) >> 4); sy <= Math.min((column.sections?.length ?? 24) - 1, (c.y + radius - minY) >> 4); sy++) {
        const section = column.sections?.[sy]
        if (!section) continue
        const palette = section.palette ?? (section.data && 'value' in section.data ? [section.data.value] : null)
        if (palette && !palette.some(id => states.has(id))) continue
        for (let x = Math.max(cx * 16, c.x - radius); x <= Math.min(cx * 16 + 15, c.x + radius); x++) {
          for (let y = Math.max(minY + sy * 16, c.y - radius); y <= Math.min(minY + sy * 16 + 15, c.y + radius); y++) {
            for (let z = Math.max(cz * 16, c.z - radius); z <= Math.min(cz * 16 + 15, c.z + radius); z++) {
              const position = new Vec3(x, y, z)
              if (position.distanceTo(c) <= radius && states.has(world.getBlockStateId(position))) found.push(position)
            }
          }
        }
      }
    }
  }
  return found
}

// Everything digVerdict needs that does not depend on the cell, computed
// once: hostiles in sight and spawners near anything the scan may test.
function stoneContext (world, { botPosition, spawn, table = null, entities = [], startY = null, scanRadius = FAN_RANGE, pickaxeType = WOODEN_PICKAXE }) {
  return {
    botPosition, spawn, table, startY, pickaxeType,
    spawners: findBlocksByName(world, botPosition, scanRadius + SPAWNER_RADIUS, SPAWNER_BLOCKS),
    hostiles: visibleHostiles(world, botPosition, entities)
  }
}

function refuse (reason, block, neighbourBlock = null, extra = {}) {
  return { ok: false, reason, block: block || null, neighbour: neighbourBlock || null, ...extra }
}

// The one safety check for every dig, MINE_STONE's and DIG_DOWN's (section
// 4.3). ctx: botPosition, spawn, table (or null), spawners and hostiles (from
// stoneContext), purpose 'mine' or 'stair', pickaxeType. The checks run in
// the order of DIG_REFUSALS (an unloaded neighbour refuses as 'unloaded' at
// the liquid check); checks that read hidden blocks only ever refuse.
function digVerdict (world, cell, ctx) {
  if (!ctx || !ctx.botPosition || !ctx.spawn) throw new Error('digVerdict needs the bot position and the spawn')
  if (!Array.isArray(ctx.spawners) || !Array.isArray(ctx.hostiles)) throw new Error('digVerdict needs precomputed spawners and hostiles')
  const purpose = ctx.purpose || 'mine'
  if (!['mine', 'stair'].includes(purpose)) throw new Error('digVerdict purpose must be mine or stair')
  const block = world.getBlock(cell)
  if (!block) return refuse('unloaded', null)
  if (AIR_BLOCKS.includes(block.name)) return refuse('nothing_to_dig', block)
  if (block.diggable === false || !(block.hardness >= 0)) return refuse('undiggable', block)
  const table = ctx.table
  if (table && cell.x === table.x && cell.z === table.z && (cell.y === table.y || cell.y === table.y - 1)) return refuse('own_table', block)
  if (inHitboxColumns(ctx.botPosition, cell)) return refuse('own_column', block)
  if (!withinRadius(ctx.spawn, { x: cell.x + 0.5, z: cell.z + 0.5 }, TARGET_MARGIN)) return refuse('outside_margin', block)
  if (isLiquidBlock(block)) return refuse('liquid_adjacent', block, block, { face: 'self' })
  for (const face of Object.keys(FACE_OFFSETS)) {
    const next = world.getBlock(neighbour(cell, face))
    if (!next) return refuse('unloaded', block, null, { face })
    if (isLiquidBlock(next)) return refuse('liquid_adjacent', block, next, { face })
  }
  let chain = 0
  let top = null
  for (let dy = 1; dy <= GRAVITY_SCAN; dy++) {
    const above = world.getBlock(cell.offset(0, dy, 0))
    if (!above || !isGravityBlock(above.name)) break
    chain++
    top = top || above
  }
  if (chain) return refuse('gravity_above', block, top, { chain })
  const below = world.getBlock(cell.offset(0, -1, 0))
  if (below?.name === 'pointed_dripstone') return refuse('dripstone_below', block, below)
  if (isIce(block.name)) return refuse('melts_to_water', block)
  if (isInfested(block.name)) return refuse('infested', block)
  const spawner = ctx.spawners.find(position => position.distanceTo(cell) <= SPAWNER_RADIUS)
  if (spawner) return refuse('spawner_nearby', block, world.getBlock(spawner))
  if (ctx.hostiles.some(entity => entity.position.distanceTo(ctx.botPosition) <= HOSTILE_RADIUS)) return refuse('hostile_nearby', block)
  if (purpose === 'stair') {
    if (isStoneSource(block.name)) return refuse('stone_source', block)
    if (isLogLike(block.name)) return refuse('log', block)
    if (isLeaves(block.name)) return refuse('leaves', block)
    if (isOre(block.name)) return refuse('ore', block)
    if (interactiveSupport(block.name)) return refuse('container', block)
    if (!digClass(block.name)) return refuse('not_dig_class', block)
  }
  const tool = chooseTool(block, { pickaxeType: ctx.pickaxeType ?? WOODEN_PICKAXE })
  if (!tool.harvestable || !(tool.ms <= MAX_DIG_MS)) return refuse('wrong_tool', block, null, { tool: tool.tool, ms: tool.ms })
  return { ok: true, reason: null, block, neighbour: null, tool: tool.tool, type: tool.type, ms: tool.ms }
}

// Where the drop of a block dug at `cell` comes to rest: the collision top of
// the first block with a shape below it. Null when the fall meets a liquid or
// an unloaded cell, or is longer than `maxFall`.
function dropRestY (world, cell, maxFall = 16) {
  for (let depth = 1; depth <= maxFall; depth++) {
    const block = world.getBlock(cell.offset(0, -depth, 0))
    if (!block || isLiquidBlock(block)) return null
    if (Array.isArray(block.shapes) && block.shapes.length) return block.position.y + collisionTop(block)
  }
  return null
}
// A drop is collectable when it rests between feet - 1 (one reversible step
// down) and feet + 2 (the vanilla pickup box, nodeCollectsItem).
function inDropBand (restY, feetY) {
  return Number.isFinite(restY) && restY >= feetY - 1 - 1e-9 && restY <= feetY + 2 + 1e-9
}

// The stone blocks the bot can mine from where it stands, best first. Both
// the nearby_stone observation and MINE_STONE call this one function.
function reachableStone (world, ctx, { litOnly = false } = {}) {
  const bot = ctx.botPosition
  const eyes = eyesAt(bot)
  const feet = new Vec3(Math.floor(bot.x), Math.floor(bot.y), Math.floor(bot.z))
  const found = []
  const r = REACH_SCAN_RADIUS
  for (let dx = -r; dx <= r; dx++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dz = -r; dz <= r; dz++) {
        if (dx * dx + dy * dy + dz * dz > r * r) continue
        const position = feet.offset(dx, dy, dz)
        const block = world.getBlock(position)
        if (!block || !isStoneSource(block.name)) continue
        const distance = position.offset(0.5, 0.5, 0.5).distanceTo(eyes)
        if (distance > REACH_DISTANCE || inHitboxColumns(bot, position)) continue
        const faces = openFaces(world, block, { litOnly })
        if (!faces.length) continue
        const restY = dropRestY(world, position)
        if (!inDropBand(restY, bot.y)) continue
        const verdict = digVerdict(world, position, { ...ctx, purpose: 'mine' })
        if (!verdict.ok || !canSeeFace(world, eyes, block, { faces })) continue
        found.push({ block, position, distance, rest_y: restY, faces, drop: stoneDrop(block.name), dig_ms: verdict.ms })
      }
    }
  }
  return found.sort((a, b) => a.distance - b.distance || a.position.y - b.position.y || a.position.x - b.position.x || a.position.z - b.position.z)
}

// The first block each ray of the fan meets, with the same per-shape
// occlusion as canSeeFace: one hit per block, at its nearest distance.
// `stats.reads` counts the loaded cells the rays visited.
function rayFan (world, eyes, { headings = FAN_HEADINGS, elevations = FAN_ELEVATIONS, range = FAN_RANGE } = {}) {
  const hits = new Map()
  const stats = { rays: 0, reads: 0 }
  for (let h = 0; h < headings; h++) {
    const yaw = 2 * Math.PI * h / headings
    for (const elevation of elevations) {
      const pitch = elevation * Math.PI / 180
      const direction = new Vec3(Math.cos(pitch) * Math.cos(yaw), Math.sin(pitch), Math.cos(pitch) * Math.sin(yaw))
      stats.rays++
      let hit = null
      const block = world.raycast(eyes, direction, range, (candidate, iterator) => {
        stats.reads++
        hit = shapeHit(candidate, iterator, eyes, direction, range)
        return hit !== null
      })
      if (!block || !hit) continue
      const key = cellKey(block.position)
      if (!hits.has(key) || hit.t < hits.get(key).t) hits.set(key, { block, t: hit.t, face: RAY_FACES[hit.face] ?? null })
    }
  }
  return { hits: [...hits.values()], stats }
}

// SEARCH_STONE's targets (stone_visible): first hits of the fan that are
// stone sources inside the margin, pass digVerdict and are not blocked in the
// memory; with litOnly, only when the face the ray met has a lit open cell.
function visibleStoneTargets (world, ctx, { memory = null, litOnly = false, ...fan } = {}) {
  const started = process.hrtime.bigint()
  const { hits, stats } = rayFan(world, eyesAt(ctx.botPosition), fan)
  const targets = []
  for (const { block, t, face } of hits) {
    if (!isStoneSource(block.name)) continue
    if (!withinRadius(ctx.spawn, { x: block.position.x + 0.5, z: block.position.z + 0.5 }, TARGET_MARGIN)) continue
    if (litOnly && !(face && lightAt(world, neighbour(block.position, face)) > 0)) continue
    if (!digVerdict(world, block.position, { ...ctx, purpose: 'mine' }).ok) continue
    if (memory && memory.blocked(block)) continue
    targets.push({ block, position: block.position, distance: t, face })
  }
  targets.sort((a, b) => a.distance - b.distance)
  return { targets, stats: { ...stats, hits: hits.length, ms: Number(process.hrtime.bigint() - started) / 1e6 } }
}

// SearchMemory's rules for stone targets (keyed by column), plus read-only
// queries so an observation never advances the attempt counter.
class StoneMemory extends SearchMemory {
  blocked (block) { return (this.blockedUntil.get(this.key(block)) || 0) > this.attempt + 1 }
  // What next() would return now, without changing anything.
  peek (candidates) {
    if (this.consecutiveBlocked >= 2) return null
    return candidates.find(block => !this.blocked(block)) || null
  }
}

// A read-only view of a world in which some cells are already dug (air).
// The staircase planner tests later checks against it, so it stays pure.
class ClearedView {
  constructor (world, positions) {
    this.world = world
    this.cleared = new Set(positions.map(cellKey))
  }

  getBlock (position) {
    const block = this.world.getBlock(position)
    const p = new Vec3(Math.floor(position.x), Math.floor(position.y), Math.floor(position.z))
    if (!block || !this.cleared.has(cellKey(p))) return block
    const air = Block.fromStateId(AIR_STATE, 0)
    air.position = p
    air.light = block.light
    air.skyLight = block.skyLight
    return air
  }

  getBlockStateId (position) { return this.getBlock(position)?.stateId ?? 0 }
  getSkyLight (position) { return typeof this.world.getSkyLight === 'function' ? this.world.getSkyLight(position) : 0 }
  getBlockLight (position) { return typeof this.world.getBlockLight === 'function' ? this.world.getBlockLight(position) : 0 }
  getColumn () { return null }

  // prismarine-world's WorldSync.raycast, reading this view.
  raycast (from, direction, range, matcher = null) {
    const iterator = new RaycastIterator(from, direction, range)
    let position = from
    while (position) {
      const block = this.getBlock(new Vec3(position.x, position.y, position.z))
      if (block) {
        if (matcher) {
          if (matcher(block, iterator)) return block
        } else {
          const intersection = iterator.intersect(block.shapes, block.position)
          if (intersection) { block.face = intersection.face; block.intersect = intersection.pos; return block }
        }
      }
      position = iterator.next()
    }
    return null
  }
}
function clearedView (world, positions) { return new ClearedView(world, positions) }

function liquidBeside (world, position) {
  for (const face of Object.keys(FACE_OFFSETS)) {
    const next = world.getBlock(neighbour(position, face))
    if (isLiquidBlock(next)) return next
  }
  return null
}

// One staircase step from `feet` (the integer cell of the bot's feet) along
// `heading` (section 4.4): dig H, M and L of the next column top first, then
// move down onto its floor. Stops early, without moving, when the next cell
// down is a stone source, so the step only exposes it (move: false). ctx is
// digVerdict's plus startY (the stage-2 start height) and reversible(view,
// from, to) -> { down, up }, the movement rules' verdict on the dug world.
function planStairStep (world, feet, heading, ctx) {
  if (!['x', 'y', 'z'].every(axis => Number.isInteger(feet?.[axis]))) throw new Error('feet must be an integer block position')
  if (!STAIR_HEADINGS[heading]) throw new Error(`unknown heading ${heading}`)
  if (!Number.isFinite(ctx?.startY)) throw new Error('planStairStep needs the stage start height')
  if (typeof ctx.reversible !== 'function') throw new Error('planStairStep needs a reversibility check')
  const [dx, dz] = STAIR_HEADINGS[heading]
  const column = new Vec3(feet.x + dx, feet.y, feet.z + dz)
  const base = { heading, column }
  const fail = (reason, extra = {}) => ({ ok: false, reason, ...base, ...extra })
  const vctx = { ...ctx, botPosition: ctx.botPosition || centreOf(feet), purpose: 'stair' }
  if (!withinRadius(ctx.spawn, { x: column.x + 0.5, z: column.z + 0.5 }, TARGET_MARGIN)) return fail('outside_margin')
  if (ctx.table && Math.max(Math.abs(column.x - ctx.table.x), Math.abs(column.z - ctx.table.z)) <= 1) return fail('near_own_table')
  const landing = column.offset(0, -1, 0)
  if (landing.y < ctx.startY - MAX_STAIR_DEPTH) return fail('depth_limit')
  const cells = [column.offset(0, 1, 0), column, landing]
  const floorPosition = column.offset(0, -2, 0)
  const blocks = cells.map(cell => world.getBlock(cell))
  const floor = world.getBlock(floorPosition)
  if (blocks.some(block => !block) || !floor) return fail('unloaded')
  const solid = blocks.slice(0, 2).filter(block => block.boundingBox === 'block').length
  let view = world
  const digs = []
  let exposed = null
  for (let i = 0; i < cells.length; i++) {
    const block = view.getBlock(cells[i])
    if (isStoneSource(block.name)) { exposed = block; break }
    if (isLiquidBlock(block)) return fail('liquid_adjacent', { cell: cells[i], block: block.name })
    if (passableCell(block)) continue
    const verdict = digVerdict(view, cells[i], vctx)
    if (!verdict.ok) return fail(verdict.reason, { cell: cells[i], block: block.name, neighbour: verdict.neighbour?.name ?? null })
    digs.push({ position: cells[i], name: block.name, tool: verdict.tool, ms: verdict.ms })
    view = clearedView(view, [cells[i]])
  }
  const digMs = digs.reduce((total, dig) => total + dig.ms, 0)
  if (digMs > MAX_STEP_DIG_MS) return fail('dig_time', { dig_ms: digMs })
  if (exposed) {
    // Nothing to dig: that stone is beside the head cell and would already be in reach if it could be mined.
    if (!digs.length) return fail('stone_source', { cell: exposed.position, block: exposed.name })
    const verdict = digVerdict(view, exposed.position, { ...vctx, purpose: 'mine' })
    if (!verdict.ok) return fail('stone_source', { cell: exposed.position, block: exposed.name, exposed_refusal: verdict.reason })
    return { ok: true, ...base, digs, dig_ms: digMs, move: false, exposes: exposed.position, exposed_block: exposed.name, solid }
  }
  if (!walkableFloor(floor)) return fail('cave_below', { floor: floor.name })
  for (const cell of [landing, column]) {
    if (!passableCell(view.getBlock(cell))) return fail('landing_blocked', { cell })
    const liquid = liquidBeside(view, cell)
    if (liquid) return fail('liquid_adjacent', { cell, neighbour: liquid.name })
  }
  const moves = ctx.reversible(view, feet, landing)
  if (!moves?.down || !moves?.up) return fail('not_reversible', { moves })
  return { ok: true, ...base, digs, dig_ms: digMs, move: true, landing, floor: { name: floor.name, position: floorPosition }, solid }
}

// The heading for the next staircase: among headings whose first step
// passes, the most solid cells at H and M (digging into a slope), ties in
// the fixed order rotated by `rotation`, like explorationIndex.
function chooseStairHeading (world, feet, ctx, rotation = 0) {
  const names = Object.keys(STAIR_HEADINGS)
  const order = names.map((_, i) => names[(i + rotation) % names.length])
  const plans = order.map(heading => planStairStep(world, feet, heading, ctx))
  const refused = Object.fromEntries(plans.filter(plan => !plan.ok).map(plan => [plan.heading, plan.reason]))
  let best = null
  for (const plan of plans) if (plan.ok && (!best || plan.solid > best.solid)) best = plan
  return best ? { ok: true, heading: best.heading, plan: best, refused, order } : { ok: false, reason: 'all_headings_refused', refused, order }
}

// Successive steps along one heading on a simulated world (the survey's
// diagnostics and the tests). stopWhen(view, feet) ends the stair after a
// move, for example when reachableStone finds a target.
function planStaircase (world, feet, heading, ctx, { maxSteps = MAX_STAIR_STEPS, stopWhen = null } = {}) {
  let view = world
  let at = feet
  const steps = []
  const result = stopped => ({ steps, stopped, feet: at, view, dig_counts: steps.filter(step => step.ok).map(step => step.digs.length) })
  for (let i = 0; i < maxSteps; i++) {
    const plan = planStairStep(view, at, heading, { ...ctx, botPosition: centreOf(at) })
    steps.push(plan)
    if (!plan.ok) return result(plan.reason)
    view = clearedView(view, plan.digs.map(dig => dig.position))
    if (!plan.move) return result('stone_exposed')
    at = plan.landing
    if (stopWhen && stopWhen(view, at)) return result('stone_exposed')
  }
  return result('step_limit')
}

// The stricter movement rules for pickups and staircase moves (section 4.5):
// the wood stage's safeMovements with a feet drop of at most one block and
// pointed dripstone avoided. The wood stage keeps its own object unchanged.
function strictMovements (bot, Movements, spawn) {
  const movements = safeMovements(bot, Movements, spawn)
  movements.maxDropDown = 2
  const dripstone = bot.registry.blocksByName.pointed_dripstone?.id
  if (Number.isInteger(dripstone)) movements.blocksToAvoid.add(dripstone)
  return movements
}

// planStairStep's reversibility check built on the real pathfinder rules,
// reading the simulated world: from `from` a drop-down move must reach `to`,
// and from `to` a move must lead back to `from`, with no digging and no
// placing (remainingBlocks 0). build(bot) returns the Movements to use.
function stairReversibility (build, Move) {
  let view = null
  const bot = {
    registry, game: { minY: -64 }, entity: { position: new Vec3(0, 0, 0), effects: {} }, entities: {}, inventory: { items: () => [] },
    blockAt: position => view.getBlock(position)
  }
  const movements = build(bot)
  const reaches = (a, b) => movements.getNeighbors(new Move(a.x, a.y, a.z, 0, 0)).some(move => move.x === b.x && move.y === b.y && move.z === b.z)
  return (world, from, to) => {
    view = world
    bot.entity.position = centreOf(from)
    const down = reaches(from, to)
    bot.entity.position = centreOf(to)
    return { down, up: reaches(to, from) }
  }
}

// Fill a standard stone pickaxe recipe (prismarine-recipe shape) with the
// stone materials actually held, mixing them if needed, as craftingRecipe
// does for planks. Null unless every slot can be filled.
function fillStoneRecipe (template, items, nameOf = id => registry.items[id]?.name) {
  if (!template?.inShape) return null
  const available = new Map()
  for (const held of items) available.set(held.type, (available.get(held.type) || 0) + held.count)
  let possible = true
  const inShape = template.inShape.map(row => row.map(ingredient => {
    if (!ingredient || ingredient.id === -1) return ingredient && { ...ingredient }
    let id = ingredient.id
    if (isStoneMaterial(nameOf(id))) id = [...available.keys()].find(type => available.get(type) > 0 && isStoneMaterial(nameOf(type)))
    if (id === undefined || (available.get(id) || 0) < 1) possible = false
    else available.set(id, available.get(id) - 1)
    return { ...ingredient, id }
  }))
  return possible ? { ...template, inShape } : null
}

module.exports = {
  STONE_SOURCES, STONE_DROPS, STONE_MATERIALS, STONE_LOOKALIKES, HAND_DIG_BLOCKS, PICKAXE_DIG_BLOCKS, GRAVITY_BLOCKS, LIQUID_BLOCKS,
  ICE_BLOCKS, SPAWNER_BLOCKS, STONE_PICKAXE_REQUIREMENT, WOODEN_PICKAXE, GRAVITY_SCAN, SPAWNER_RADIUS, HOSTILE_RADIUS, HOSTILE_STOP_RADIUS,
  MAX_DIG_MS, MAX_STEP_DIG_MS, MAX_STAIR_DEPTH, MAX_STAIR_STEPS, REACH_SCAN_RADIUS, REACH_DISTANCE, FAN_HEADINGS, FAN_ELEVATIONS,
  FAN_RANGE, STAIR_HEADINGS, FACE_OFFSETS, DIG_REFUSALS,
  isStoneSource, isStoneMaterial, stoneDrop, isInfested, isGravityBlock, isIce, isOre, isLogLike, digClass, isLiquidBlock, openCell,
  isHostile, stoneInventoryCounts, pickaxeBlock, digTimeMs, chooseTool, inHitboxColumns, lightAt, openFaces, clearLine, visibleHostiles,
  findBlocksByName, stoneContext, digVerdict, dropRestY, inDropBand, reachableStone, rayFan, visibleStoneTargets, StoneMemory,
  ClearedView, clearedView, planStairStep, chooseStairHeading, planStaircase, strictMovements, stairReversibility, fillStoneRecipe
}
