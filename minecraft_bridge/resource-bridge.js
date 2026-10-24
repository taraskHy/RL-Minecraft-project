'use strict'

// Natural-world executor for learned macro-actions: the eight wood actions, and
// under --goal stone_pickaxe the stone stage's five. This file never sends
// chat commands, creates resources, resets a world, or chooses the next action.
const readline = require('node:readline')
const { Vec3 } = require('vec3')

const ACTIONS = ['SEARCH', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE', 'PLACE_TABLE', 'APPROACH_TABLE', 'CRAFT_PICKAXE',
  'SEARCH_STONE', 'DIG_DOWN', 'MINE_STONE', 'RETURN_TO_TABLE', 'CRAFT_STONE_PICKAXE']
// The default goal accepts only the wood actions, as before the stone stage.
const WOOD_ACTIONS = ACTIONS.slice(0, 8)
const GOALS = ['wooden_pickaxe', 'stone_pickaxe']
// Decision 9 takes its declared fallback: the lit-face rule is off.
// Mineflayer keeps light only from map_chunk and update_light and never
// recomputes it, and the vanilla 1.21.4 server sends update_light only for
// chunks on the edge of a player's view (ChunkHolder.broadcastChanges), so
// every cell the bot digs keeps light 0 and stone exposed by digging would
// never count. Stone in dark places therefore counts as seen and in reach
// (disclosed). The helpers' litOnly option stays, tested and unused.
const STONE_LIT_ONLY = false
// The passive survey simulates this many staircase steps per heading and
// scans this far down the bot's column for stone.
const SURVEY_STAIR_STEPS = 6
const SURVEY_DEPTH = 64
const BACKEND = 'minecraft-java-natural-resources'
const MAX_RADIUS = 64
// Navigation targets' block coordinates stay this far inside MAX_RADIUS (a
// target cell's centre about 2.3 blocks); together with STEP_MARGIN, a goal
// tolerance or an item pickup beside a target cannot carry the bot across.
const TARGET_MARGIN = 3
// The pathfinder walks only through cells whose centre is this far inside.
const STEP_MARGIN = 1.5
const SEARCH_RADIUS = 32
// One request may take 44 seconds; SEARCH keeps a margin for its two observations.
const SEARCH_BUDGET_MS = 36000
const CANOPY_BUDGET_MS = 20000
const MAX_CANOPY_LAYERS = 6
const CANOPY_SCAN_DEPTH = 12
// Leaf blocks a foliage escape may clear beside the bot in one decision.
const MAX_ESCAPE_LEAVES = 2
// Spawn-relative exploration rings of 8, 16, ... blocks before starting over.
const EXPLORATION_RINGS = 6
const EXPLORATION_DIRECTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
// A descent must never fall through or land on these. Liquids, bushes and
// powder snow have empty bounding boxes, so solidity alone cannot exclude them;
// a big dripleaf is solid but tilts and drops whoever lands on it.
const FALL_HAZARDS = ['water', 'lava', 'fire', 'soul_fire', 'cactus', 'magma_block', 'powder_snow', 'sweet_berry_bush',
  'campfire', 'soul_campfire', 'pointed_dripstone', 'cobweb', 'bubble_column', 'big_dripleaf']
// Vanilla collects a dropped item when the item's box meets the player's box
// inflated by one block horizontally: 0.3 + 1.0 + 0.125 = 1.425 per axis. The
// pathfinder may stop up to 0.2 from a node centre, so a pickup goal accepts
// only nodes whose centre lies within 1.2 of the item on both axes.
const PICKUP_REACH = 1.2
class ObservationError extends Error {}

class SearchMemory {
  constructor () { this.attempt = 0; this.blockedUntil = new Map(); this.consecutiveBlocked = 0 }
  // Keyed by trunk column: which log of a trunk is visible changes with small
  // movements, and a blocked trunk must stay blocked whichever log is seen.
  key (block) { return `${block.position.x},${block.position.z}` }
  next (candidates) {
    this.attempt++
    // A local obstruction should lead to a new view, not repeated attempts at
    // every upper block in the same inaccessible tree.
    if (this.consecutiveBlocked >= 2) { this.consecutiveBlocked = 0; return null }
    return candidates.find(block => (this.blockedUntil.get(this.key(block)) || 0) <= this.attempt) || null
  }
  failed (block) { this.blockedUntil.set(this.key(block), this.attempt + 8); this.consecutiveBlocked++ }
  reached () { this.consecutiveBlocked = 0 }
}

function replaceableForTable (name) {
  return ['air', 'cave_air', 'short_grass', 'tall_grass', 'fern', 'large_fern', 'dead_bush', 'snow'].includes(name)
}

function isLeaves (name) {
  return typeof name === 'string' && name.endsWith('_leaves')
}

// Vanilla fall damage without effects or enchantments: ceil(distance - 3).
function fallDamage (blocks) {
  if (!Number.isFinite(blocks) || blocks < 0) throw new Error('fall distance must be a nonnegative number')
  return Math.max(0, Math.ceil(blocks - 3))
}

// A descent step is acceptable when it costs no health, or at most two points
// while health stays above the learning state's ten-point bin.
function descentAcceptable (fall, health) {
  const damage = fallDamage(fall)
  return damage === 0 || (damage <= 2 && health - damage > 10)
}

// Decide whether clearing the leaf block under the feet gives a verified
// landing. `column` lists the blocks below the feet, nearest first, as
// { name, boundingBox, top } or null when unloaded. `top` is the collision
// height inside the block cell (1 for a full cube, 0.0625 for a carpet): the
// fall ends on that surface, not on the cell top. Pure, so tests can cover it.
function planCanopyDescent ({ feetY, health, column }) {
  if (!Number.isInteger(feetY)) throw new Error('feetY must be an integer block coordinate')
  if (!Number.isFinite(health) || health < 0 || health > 20) throw new Error('health must be a number from 0 to 20')
  if (!Array.isArray(column) || column.length < 2) throw new Error('column must list the blocks below the feet')
  const support = column[0]
  if (!support || !isLeaves(support.name) || support.boundingBox !== 'block') {
    return { ok: false, reason: 'not_standing_on_leaves', support: support?.name ?? null }
  }
  for (let index = 1; index < column.length; index++) {
    const block = column[index]
    const y = feetY - 1 - index
    if (!block) return { ok: false, reason: 'unloaded_column', y }
    if (FALL_HAZARDS.includes(block.name)) return { ok: false, reason: 'hazard_below', hazard: block.name, y }
    if (block.boundingBox === 'block') {
      const top = Number.isFinite(block.top) ? Math.min(1, Math.max(0, block.top)) : 1
      const landingY = y + top
      const fall = feetY - landingY
      const damage = fallDamage(fall)
      if (!descentAcceptable(fall, health)) return { ok: false, reason: 'fall_too_far', fall, expected_damage: damage, landing_block: block.name }
      return { ok: true, dig_block: support.name, landing_y: landingY, landing_block: block.name, fall, expected_damage: damage }
    }
  }
  return { ok: false, reason: 'no_landing_within_scan', scanned: column.length - 1 }
}

// Whether a bot standing on path node `node` collects an item lying at `item`:
// horizontal centre distance under PICKUP_REACH on both axes, and the item's
// box inside the player's box inflated by half a block vertically.
function nodeCollectsItem (node, item) {
  const dy = item.y - node.y
  return Math.abs(node.x + 0.5 - item.x) < PICKUP_REACH && Math.abs(node.z + 0.5 - item.z) < PICKUP_REACH && dy > -0.75 && dy < 2.3
}

// Choose where a bot walled in on all four sides can clear leaf blocks to walk
// out. `neighbours` maps a direction key to { feet, head, below, lower }: the
// cells at feet and head height beside the bot and the two blocks under that
// column, each { name, boundingBox } or null when unloaded or out of bounds.
// Only leaf blocks may be cleared, the exit needs a solid non-hazard floor at
// most one block down, and the direction needing the fewest blocks wins. Pure.
function planLeafEscape (neighbours) {
  const options = []
  for (const [direction, n] of Object.entries(neighbours || {})) {
    if (!n || !n.feet || !n.head || !n.below) continue
    const cells = [n.feet, n.head]
    if (!cells.every(cell => cell.boundingBox === 'empty' || isLeaves(cell.name))) continue
    if (cells.some(cell => FALL_HAZARDS.includes(cell.name))) continue
    let floor = null
    if (n.below.boundingBox === 'block') floor = n.below
    else if (n.below.boundingBox === 'empty' && !FALL_HAZARDS.includes(n.below.name) && n.lower?.boundingBox === 'block') floor = n.lower
    if (!floor || FALL_HAZARDS.includes(floor.name)) continue
    const dig = cells.filter(cell => isLeaves(cell.name)).map(cell => cell.name)
    if (!dig.length || dig.length > MAX_ESCAPE_LEAVES) continue
    options.push({ direction, dig, floor: floor.name })
  }
  if (!options.length) return { ok: false, reason: 'no_leaf_exit' }
  options.sort((a, b) => a.dig.length - b.dig.length)
  return { ok: true, ...options[0] }
}

// Spawn-relative exploration target for the n-th long move: every heading of
// a ring before the radius grows, rings of 8 to 48 blocks, then over again.
// Diagonal targets are capped so that every target, plus the goal's two-block
// tolerance, stays inside the 64-block starting radius.
function explorationTarget (index) {
  if (!Number.isInteger(index) || index < 0) throw new Error('exploration index must be a nonnegative integer')
  const [dx, dz] = EXPLORATION_DIRECTIONS[index % EXPLORATION_DIRECTIONS.length]
  const ring = Math.floor(index / EXPLORATION_DIRECTIONS.length) % EXPLORATION_RINGS
  return { dx, dz, radius: Math.min(8 * (ring + 1), Math.floor((MAX_RADIUS - TARGET_MARGIN) / Math.hypot(dx, dz))) }
}

const EYE_HEIGHT = 1.62
const HITBOX_HALF_WIDTH = 0.3
const PLACEMENT_REACH = 4.3
const PLACEMENT_FORBIDDEN_SUPPORTS = ['crafting_table', 'magma_block', 'cactus', 'campfire', 'soul_campfire']
// Blocks a click uses instead of placing against (the bot never sneaks):
// containers, beds, doors, trapdoors, workstations. The pathfinder's own list
// plus the 1.21 blocks it lacks.
const INTERACTIVE_SUPPORTS = new Set([...require('mineflayer-pathfinder/lib/interactable.json'), 'lectern', 'loom', 'grindstone',
  'cake', 'decorated_pot', 'crafter', 'chiseled_bookshelf', 'cartography_table', 'beehive', 'bee_nest', 'respawn_anchor', 'vault', 'trial_spawner'])
function interactiveSupport (name) {
  return INTERACTIVE_SUPPORTS.has(name) || name.endsWith('_cake') || name.endsWith('shulker_box')
}
// Orthogonal neighbours first: a table beside the bot then shows a whole face.
const APPROACH_OFFSETS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
// Face sample fractions, centre first so a plainly visible block costs one ray.
const SAMPLE_FRACTIONS = [0.5, 0.1, 0.9]

// Whether a block lies below the bot's feet (or holds them, on a partial
// block) in a column that carries any part of its 0.6-wide hitbox. CHOP and
// SEARCH never target such a block; only the canopy descent removes the leaf
// block under the feet. Chopping its own support dropped the bot into a
// two-by-two trunk on the PC.
function underHitbox (botPosition, blockPosition) {
  if (!(blockPosition.y < botPosition.y - 1e-6)) return false
  return [[0, 0], [HITBOX_HALF_WIDTH, HITBOX_HALF_WIDTH], [HITBOX_HALF_WIDTH, -HITBOX_HALF_WIDTH], [-HITBOX_HALF_WIDTH, HITBOX_HALF_WIDTH], [-HITBOX_HALF_WIDTH, -HITBOX_HALF_WIDTH]].some(([dx, dz]) =>
    Math.floor(botPosition.x + dx) === blockPosition.x && Math.floor(botPosition.z + dz) === blockPosition.z)
}

function withinRadius (spawn, position, margin = 0) {
  return Math.hypot(position.x - spawn.x, position.z - spawn.z) <= MAX_RADIUS - margin
}

function eyesAt (position) { return new Vec3(position.x, position.y + EYE_HEIGHT, position.z) }

function escapeNote (escape) {
  const count = escape.blocks_dug.length
  return count ? `cleared_${count}_leaf_wall_block${count === 1 ? '' : 's'}` : `leaf_escape_refused:${escape.stopped || 'unknown'}`
}

// Points to aim at on a block's collision shapes: each shape's centre (unless
// only some faces count) and a 3x3 grid on every face that looks towards the
// eyes. A ray to any of them that reaches this block first means it is seen.
function visibilitySamples (block, eyes, faces = null) {
  const samples = []
  const eye = [eyes.x, eyes.y, eyes.z]
  for (const [x0, y0, z0, x1, y1, z1] of block.shapes) {
    const lo = [block.position.x + x0, block.position.y + y0, block.position.z + z0]
    const hi = [block.position.x + x1, block.position.y + y1, block.position.z + z1]
    if (!faces) samples.push(new Vec3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2))
    for (const [name, axis, plane, sign] of [['west', 0, lo[0], -1], ['east', 0, hi[0], 1], ['bottom', 1, lo[1], -1], ['top', 1, hi[1], 1], ['north', 2, lo[2], -1], ['south', 2, hi[2], 1]]) {
      if (faces && !faces.includes(name)) continue
      if ((eye[axis] - plane) * sign <= 0) continue // the face looks away from the eyes
      const [a, b] = [0, 1, 2].filter(other => other !== axis)
      for (const fa of SAMPLE_FRACTIONS) {
        for (const fb of SAMPLE_FRACTIONS) {
          const point = [0, 0, 0]
          point[axis] = plane
          point[a] = lo[a] + (hi[a] - lo[a]) * fa
          point[b] = lo[b] + (hi[b] - lo[b]) * fb
          samples.push(new Vec3(point[0], point[1], point[2]))
        }
      }
    }
  }
  return samples
}

// Line of sight from `eyes` to `block` (optionally to some of its faces only)
// through a prismarine world. Occluders are tested per collision shape, so a
// ray that enters a slab's empty upper half is not "seeing" the slab, and
// shapes behind the eyes (which the library also reports) are ignored.
function canSeeFace (world, eyes, block, { faces = null } = {}) {
  if (!block || !Array.isArray(block.shapes) || !block.shapes.length) return false
  for (const point of visibilitySamples(block, eyes, faces)) {
    const delta = point.minus(eyes)
    const range = delta.norm()
    if (range < 1e-3) return true
    const direction = delta.scaled(1 / range)
    const hit = world.raycast(eyes, direction, range + 0.05, (candidate, iterator) => {
      for (const shape of candidate.shapes || []) {
        const intersection = iterator.intersect([shape], candidate.position)
        if (!intersection) continue
        const t = intersection.pos.minus(eyes).dot(direction)
        if (t >= -1e-9 && t <= range + 0.05) return true
      }
      return false
    })
    if (hit && hit.position.equals(block.position)) return true
  }
  return false
}

// Vanilla refuses to place a block whose box meets the player's: a cell is
// overlapped when the bot's centre is within 0.8 (half a cell plus half the
// hitbox) of the cell centre on both horizontal axes.
function hitboxOverlapsCell (botPosition, cell) {
  const reach = 0.5 + HITBOX_HALF_WIDTH
  return Math.abs(cell.x + 0.5 - botPosition.x) < reach && Math.abs(cell.z + 0.5 - botPosition.z) < reach
}

// Orthogonal neighbours of the bot's cell first, then the smallest height
// change, then the nearest cell centre.
function placementRank (botPosition, cell) {
  const feet = botPosition.floored()
  const offset = Math.abs(cell.x - feet.x) + Math.abs(cell.z - feet.z)
  return [offset === 1 ? 0 : 1, Math.abs(cell.y - feet.y), Math.hypot(cell.x + 0.5 - botPosition.x, cell.z + 0.5 - botPosition.z)]
}

function compareRanks (a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i]
  return 0
}

// Cells where the bot could place its table: replaceable, on a solid support,
// not overlapped by its hitbox, inside the margin, and (when withinReach) with
// the support's top face visible and in reach. Pure over a prismarine world.
// Whether an entity box ({ x0, y0, z0, x1, y1, z1 }) strictly meets a unit cell.
function cellIntersectsBox (cell, box) {
  const e = 1e-7
  return box.x0 < cell.x + 1 - e && box.x1 > cell.x + e && box.y0 < cell.y + 1 - e && box.y1 > cell.y + e &&
    box.z0 < cell.z + 1 - e && box.z1 > cell.z + e
}

// `obstacles` are the boxes of nearby creatures (vanilla refuses a block that
// would meet one); `excludedColumns` are cells' x, z that must stay open.
function placementCandidates (world, botPosition, spawn, { radius, withinReach, obstacles = [], excludedColumns = [] }) {
  const feet = botPosition.floored()
  const eyes = eyesAt(botPosition)
  const candidates = []
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dy = -3; dy <= 3; dy++) {
        const position = feet.offset(dx, dy, dz)
        if (hitboxOverlapsCell(botPosition, position) || !withinRadius(spawn, position, TARGET_MARGIN)) continue
        if (excludedColumns.some(column => column.x === position.x && column.z === position.z)) continue
        if (obstacles.some(box => cellIntersectsBox(position, box))) continue
        const empty = world.getBlock(position)
        const support = world.getBlock(position.offset(0, -1, 0))
        if (!replaceableForTable(empty?.name) || support?.boundingBox !== 'block' || PLACEMENT_FORBIDDEN_SUPPORTS.includes(support.name) || interactiveSupport(support.name)) continue
        if (withinReach && (support.position.offset(0.5, 1, 0.5).distanceTo(eyes) > PLACEMENT_REACH || !canSeeFace(world, eyes, support, { faces: ['top'] }))) continue
        candidates.push({ position, support, rank: placementRank(botPosition, position) })
      }
    }
  }
  return candidates.sort((a, b) => compareRanks(a.rank, b.rank))
}

function standableIn (world, position) {
  const feetBlock = world.getBlock(position)
  const headBlock = world.getBlock(position.offset(0, 1, 0))
  return Boolean(feetBlock && headBlock && feetBlock.boundingBox === 'empty' && headBlock.boundingBox === 'empty' &&
    !FALL_HAZARDS.includes(feetBlock.name) && !FALL_HAZARDS.includes(headBlock.name))
}

function collisionTop (block) {
  return Array.isArray(block?.shapes) && block.shapes.length ? Math.max(...block.shapes.map(shape => shape[4])) : 0
}

// The pathfinder's classification: a feet or head cell is passable when it is
// empty or lower than 0.1 (carpets); a floor is walkable when it is solid and
// no taller than a block (fences and walls are not). The floor must also reach
// above half a block, so the bot really stands in the cell above it.
function passableCell (block) {
  return Boolean(block) && !FALL_HAZARDS.includes(block.name) && (block.boundingBox === 'empty' || collisionTop(block) < 0.1)
}
function walkableFloor (block) {
  const top = collisionTop(block)
  return Boolean(block) && block.boundingBox === 'block' && top > 0.5 && top <= 1 && !FALL_HAZARDS.includes(block.name)
}

// Cells beside the table where the bot can stand inside the margin: level
// first, then one below and one above (a table on a bump), orthogonal first.
function approachCells (world, tablePosition, spawn) {
  const cells = []
  for (const dy of [0, -1, 1]) {
    for (const [dx, dz] of APPROACH_OFFSETS) {
      const cell = tablePosition.offset(dx, dy, dz)
      if (passableCell(world.getBlock(cell)) && passableCell(world.getBlock(cell.offset(0, 1, 0))) &&
        walkableFloor(world.getBlock(cell.offset(0, -1, 0))) && withinRadius(spawn, cell, TARGET_MARGIN)) cells.push(cell)
    }
  }
  return cells
}

// Cost the pathfinder adds for stepping into a cell. Cells whose centre is
// STEP_MARGIN inside the radius are free, so a navigation cut off mid-path
// still leaves the bot inside it. A bot pushed past that line (knockback) may
// still use its own column and cells no farther out than it already is, up to
// a centre 0.75 inside the radius, so it can always step back inward.
function stepExclusion (spawn, cell, botPosition) {
  const centre = Math.hypot(cell.x + 0.5 - spawn.x, cell.z + 0.5 - spawn.z)
  if (centre <= MAX_RADIUS - STEP_MARGIN) return 0
  if (!botPosition) return 100
  const ownX = Math.floor(botPosition.x)
  const ownZ = Math.floor(botPosition.z)
  if (cell.x === ownX && cell.z === ownZ) return 0
  const ownCentre = Math.hypot(ownX + 0.5 - spawn.x, ownZ + 0.5 - spawn.z)
  return centre <= ownCentre + 1e-9 && centre <= MAX_RADIUS - 0.75 ? 0 : 100
}

// Loaded on first use, never when this module loads: stone-helpers requires it.
function stoneHelpers () { return require('./stone-helpers') }

function parseArgs (args) {
  const config = { host: '127.0.0.1', port: 25567, username: 'RLResourceBot', version: '1.21.4', goal: 'wooden_pickaxe', stoneSurvey: 'off' }
  const fields = { '--host': 'host', '--port': 'port', '--username': 'username', '--version': 'version', '--goal': 'goal', '--stone-survey': 'stoneSurvey' }
  for (let i = 0; i < args.length; i++) {
    if (!Object.hasOwn(fields, args[i]) || args[i + 1] === undefined) throw new Error(`Unknown or incomplete option: ${args[i]}`)
    config[fields[args[i]]] = args[++i]
  }
  config.port = Number(config.port)
  if (!['127.0.0.1', 'localhost', '::1'].includes(config.host)) throw new Error('Natural-resource bridge only connects to localhost')
  if (!Number.isInteger(config.port) || config.port < 1 || config.port > 65535) throw new Error('Invalid port')
  if (!/^[A-Za-z0-9_]{1,16}$/.test(config.username)) throw new Error('Invalid bot username')
  if (config.version !== '1.21.4') throw new Error('Natural-resource bridge is pinned to Minecraft Java 1.21.4')
  // The goal decides success and which actions run; the survey only adds diagnostics.
  if (!GOALS.includes(config.goal)) throw new Error(`--goal must be one of ${GOALS.join(', ')}`)
  if (!['on', 'off'].includes(config.stoneSurvey)) throw new Error('--stone-survey must be on or off')
  config.stoneSurvey = config.stoneSurvey === 'on'
  return config
}

function actionsFor (goal) {
  if (goal === 'wooden_pickaxe') return WOOD_ACTIONS
  if (goal === 'stone_pickaxe') return ACTIONS
  throw new Error(`Unknown goal: ${goal}`)
}

function validateAction (action, goal = 'wooden_pickaxe') {
  const count = actionsFor(goal).length
  if (!Number.isInteger(action) || action < 0 || action >= count) throw new Error(`action must be an integer from 0 through ${count - 1}`)
  return action
}

// The stone actions' helpers arrive in step S3. Until then a stone action
// stops the episode with this error, never an observed transition that would
// look like an ordinary refusal.
function runnableAction (action, goal) {
  validateAction(action, goal)
  if (action >= WOOD_ACTIONS.length) throw new Error(`${ACTIONS[action]} (action ${action}) is not implemented until step S3; this episode is invalid`)
  return action
}

// The wood goal ends on any wooden pickaxe in the inventory, as it always has;
// the stone goal only on a held stone pickaxe (section 4.8).
function goalReached (goal, counts, heldItem) {
  if (goal === 'wooden_pickaxe') return counts.pickaxes > 0
  if (goal === 'stone_pickaxe') return heldItem === 'stone_pickaxe'
  throw new Error(`Unknown goal: ${goal}`)
}

function inventoryCounts (inventory) {
  const counts = { logs: 0, planks: 0, sticks: 0, tables: 0, pickaxes: 0 }
  for (const [name, amount] of Object.entries(inventory)) {
    if (!Number.isInteger(amount) || amount < 0) throw new Error('Invalid inventory quantity')
    if (name.endsWith('_log')) counts.logs += amount
    if (name.endsWith('_planks')) counts.planks += amount
    if (name === 'stick') counts.sticks += amount
    if (name === 'crafting_table') counts.tables += amount
    if (name === 'wooden_pickaxe') counts.pickaxes += amount
  }
  return { ...counts, ...stoneHelpers().stoneInventoryCounts(inventory) }
}

function validatePhysicalState (observation) {
  if (!observation.position || !['x', 'y', 'z'].every(axis => Number.isFinite(observation.position[axis]))) throw new Error('Incomplete or non-finite position')
  if (!Number.isFinite(observation.health) || observation.health < 0 || observation.health > 20) throw new Error('Incomplete or invalid health')
  if (!Number.isFinite(observation.food) || observation.food < 0 || observation.food > 20) throw new Error('Incomplete or invalid food')
  if (observation.gamemode !== 'survival') throw new Error('Resource bot must remain in Survival mode')
  return observation
}

function craftingRecipe (bot, item, table) {
  const ordinary = bot.recipesFor(item.id, null, 1, table)[0]
  if (ordinary) return ordinary
  // Minecraft accepts mixed plank species, but minecraft-data enumerates only
  // single-species variants. Fill a standard recipe with the actual inventory
  // IDs, so aggregating planks in the learning state does not hide a dead end.
  // Stone pickaxe materials are mixed the same way (decision 10).
  if (item.name === 'stone_pickaxe') return stoneHelpers().fillStoneRecipe(bot.recipesAll(item.id, null, table)[0], bot.inventory.items(), id => bot.registry.items[id]?.name)
  if (!['stick', 'crafting_table', 'wooden_pickaxe'].includes(item.name)) return null
  const template = bot.recipesAll(item.id, null, table)[0]
  if (!template?.inShape) return null
  const available = new Map()
  for (const held of bot.inventory.items()) available.set(held.type, (available.get(held.type) || 0) + held.count)
  let possible = true
  const inShape = template.inShape.map(row => row.map(ingredient => {
    if (ingredient.id === -1) return { ...ingredient }
    let id = ingredient.id
    if (bot.registry.items[id]?.name.endsWith('_planks')) id = [...available.keys()].find(type => available.get(type) > 0 && bot.registry.items[type]?.name.endsWith('_planks'))
    if (id === undefined || (available.get(id) || 0) < 1) possible = false
    else available.set(id, available.get(id) - 1)
    return { ...ingredient, id }
  }))
  return possible ? { ...template, inShape } : null
}

function safeMovements (bot, Movements, spawn) {
  const movements = new Movements(bot)
  movements.canDig = false
  movements.allow1by1towers = false
  movements.scafoldingBlocks = []
  movements.allowParkour = false
  movements.allowSprinting = false
  movements.allowFreeMotion = false
  // Pathfinder compares feet Y to the supporting block's Y, so 4 means a
  // maximum three-block feet-height descent, not a four-block fall. Larger
  // drops are handled only by the explicit, health-aware canopy descent.
  movements.maxDropDown = 4
  movements.infiniteLiquidDropdownDistance = false
  for (const name of ['water', 'lava', 'fire', 'soul_fire', 'cactus', 'magma_block', 'powder_snow', 'sweet_berry_bush', 'campfire', 'soul_campfire']) {
    const type = bot.registry.blocksByName[name]?.id
    if (Number.isInteger(type)) movements.blocksToAvoid.add(type)
  }
  movements.exclusionAreasStep.push(block => stepExclusion(spawn, block.position, bot.entity?.position))
  movements.exclusionAreasPlace.push(() => 100)
  return movements
}

function roundedPosition (position) {
  return { x: Math.round(position.x * 100) / 100, y: Math.round(position.y * 100) / 100, z: Math.round(position.z * 100) / 100 }
}

// Uses left on the wooden pickaxe, the held one first: its maximum durability
// less prismarine-item's durabilityUsed, which reads 1.21.4's damage
// component. Null without a wooden pickaxe or when the value is unreadable.
function pickaxeUsesLeft (items, heldItem = null) {
  const pickaxe = heldItem?.name === 'wooden_pickaxe' ? heldItem : items.find(item => item.name === 'wooden_pickaxe')
  if (!pickaxe) return null
  let used
  try { used = pickaxe.durabilityUsed } catch (_) { return null }
  return Number.isInteger(used) && used >= 0 && Number.isInteger(pickaxe.maxDurability) ? pickaxe.maxDurability - used : null
}

// A ray-fan target as the observation and the survey record it.
function fanTargetSummary (target) {
  return { name: target.block.name, x: target.position.x, y: target.position.y, z: target.position.z, distance: Math.round(target.distance * 100) / 100, face: target.face }
}

// The stone stage's observation fields (section 3.6) from one scan of the
// loaded world. `scene` is stoneContext's input, with the climb search and
// the staircase record: the context MINE_STONE and SEARCH_STONE use, so the
// state bits are those actions' preconditions.
// - nearby_stone, and stone_reach_candidates (the 3 best), from the reach
//   list MINE_STONE re-checks;
// - stone_targets: the up to 3 nearest ray-fan targets, in the order
//   SEARCH_STONE tries them (the memory's blocked targets are left out);
// - stone_visible: stone in reach, or the target the memory would give
//   next; stone_targets_withheld when targets exist but the memory gives
//   none (two failures in a row force exploration);
// - hostile_nearby: the mobs digVerdict refuses on.
function stoneScan (world, scene, { memory = null, litOnly = STONE_LIT_ONLY } = {}) {
  const stone = stoneHelpers()
  const ctx = stone.stoneContext(world, scene)
  const started = process.hrtime.bigint()
  const reach = stone.reachableStone(world, ctx, { litOnly })
  const reachMs = Number(process.hrtime.bigint() - started) / 1e6
  const fan = stone.visibleStoneTargets(world, ctx, { memory, litOnly })
  const target = memory ? memory.peek(fan.targets) : fan.targets[0] || null
  return {
    ctx,
    reach,
    fan,
    reach_ms: reachMs,
    lit_only: litOnly,
    fields: {
      nearby_stone: reach.length > 0,
      stone_visible: reach.length > 0 || Boolean(target),
      stone_targets: fan.targets.slice(0, 3).map(fanTargetSummary),
      stone_targets_withheld: fan.targets.length > 0 && !target,
      stone_reach_candidates: reach.slice(0, 3).map(candidate => ({
        name: candidate.block.name, x: candidate.position.x, y: candidate.position.y, z: candidate.position.z,
        distance: Math.round(candidate.distance * 100) / 100, drop: candidate.drop, rest_y: candidate.rest_y
      })),
      hostile_nearby: ctx.hostiles.map(entity => ({
        name: entity.name, ...roundedPosition(entity.position), distance: Math.round(entity.position.distanceTo(scene.botPosition) * 100) / 100
      }))
    }
  }
}

// The stage-2 start that depth_below_stage_start and DIG_DOWN's depth cap
// count from (section 4.4): the natural floor under the hand-off cell `feet`,
// plus one. A hand-off can stand on the bot's own table, a log or a canopy
// (5 of 124 at 87673b8), so the scan skips the own table, logs, leaves and
// passable cells, at most `maxDepth` cells down (the canopy descent's scan).
// Without such a floor (an unloaded cell, or none in range) it stays the
// hand-off y, which can only make the cap shallower. `skipped` lists the
// skipped cells as [name, count], top first. Pure; the bridge latches it once.
function stageStartReference (world, feet, table = null, maxDepth = CANOPY_SCAN_DEPTH) {
  const stone = stoneHelpers()
  const skipped = []
  const result = (y, floor, stopped) => ({ hand_off_y: feet.y, y, floor, skipped, stopped })
  for (let depth = 1; depth <= maxDepth; depth++) {
    const position = feet.offset(0, -depth, 0)
    const block = world.getBlock(position)
    if (!block) return result(feet.y, null, 'unloaded')
    const ownTable = Boolean(table) && block.name === 'crafting_table' && position.x === table.x && position.y === table.y && position.z === table.z
    if (!ownTable && !stone.isLogLike(block.name) && !isLeaves(block.name) && !passableCell(block)) return result(position.y + 1, block.name, null)
    const last = skipped[skipped.length - 1]
    if (last && last[0] === block.name) last[1]++
    else skipped.push([block.name, 1])
  }
  return result(feet.y, null, 'scan_limit')
}

// Survey only: how far below the feet the first stone source lies in the
// bot's own column, with the layers above it as [name, count], top first.
function depthToStone (world, feet, maxDepth = SURVEY_DEPTH) {
  const stone = stoneHelpers()
  const layers = []
  for (let depth = 1; depth <= maxDepth; depth++) {
    const block = world.getBlock(feet.offset(0, -depth, 0))
    if (!block) return { depth: null, block: null, layers, stopped: 'unloaded' }
    if (stone.isStoneSource(block.name)) return { depth, block: block.name, layers, stopped: null }
    const last = layers[layers.length - 1]
    if (last && last[0] === block.name) last[1]++
    else layers.push([block.name, 1])
  }
  return { depth: null, block: null, layers, stopped: 'scan_limit' }
}

// One planned staircase step in a survey record: what it digs, or why it is refused.
function stairStepSummary (step) {
  const cell = position => position ? [position.x, position.y, position.z] : null
  const summary = step.ok
    ? { ok: true, digs: step.digs.map(dig => [dig.name, dig.tool]), dig_ms: step.dig_ms, move: step.move, landing_y: step.landing?.y, floor: step.floor?.name, exposes: step.exposed_block }
    : { ok: false, reason: step.reason, cell: cell(step.cell), block: step.block, neighbour: step.neighbour, face: step.face, floor: step.floor, exposed_refusal: step.exposed_refusal, dig_ms: step.dig_ms, moves: step.moves, error: step.error }
  return Object.fromEntries(Object.entries(summary).filter(([, value]) => value !== undefined && value !== null))
}

// Survey only: what planStairStep would refuse on each of the four headings
// for up to `maxSteps` consecutive steps, simulated on a view of the world
// with the planned cells dug; nothing is dug. `ctx` is the observation's scan
// context (the stage start as startY, the climb search, the staircase
// record). Every heading is planned, the ones the first-step table rule
// would drop included, so each reports the terrain's reason. A heading ends
// at a refusal, at exposed stone, or once MINE_STONE would have a target:
// the observation's reach function and light rule, with the floors of the
// staircase so far protected (planStaircase passes it in stepCtx).
function surveyStaircases (world, feet, ctx, maxSteps = SURVEY_STAIR_STEPS) {
  const stone = stoneHelpers()
  const stopWhen = (view, at, stepCtx) => stone.reachableStone(view, stepCtx, { litOnly: STONE_LIT_ONLY }).length > 0
  return Object.fromEntries(Object.keys(stone.STAIR_HEADINGS).map(heading => {
    const stair = stone.planStaircase(world, feet, heading, ctx, { maxSteps, stopWhen })
    return [heading, { stopped: stair.stopped, dig_counts: stair.dig_counts, end_y: stair.feet.y, steps: stair.steps.map(stairStepSummary) }]
  }))
}

// The passive survey's diagnostic_only record (section 7.6, item 1), never an
// input to any state encoder. `scan` is the observation's own stoneScan, whose
// ctx carries the stage start (startY), and `stageStart` the latched
// stageStartReference. The survey adds the stage start, the depth to stone
// (also counted from the stage start), where the bot stands relative to its
// own table and which headings the first-step table rule drops, the four
// simulated staircases, the ray-fan cost, and how many targets the light rule
// removed (none while it is off).
function stoneSurveyRecord (world, feet, scan, { memory = null, stageStart = null } = {}) {
  const stone = stoneHelpers()
  const started = process.hrtime.bigint()
  const { startY, table } = scan.ctx
  const depth = depthToStone(world, feet)
  const anyLight = scan.lit_only
    ? { reach: stone.reachableStone(world, scan.ctx).length, fan: stone.visibleStoneTargets(world, scan.ctx, { memory }).targets.length }
    : { reach: scan.reach.length, fan: scan.fan.targets.length }
  const record = {
    lit_only: scan.lit_only,
    stage_start: stageStart,
    depth_to_stone: { ...depth, from_stage_start: depth.depth === null || !Number.isFinite(startY) ? null : depth.depth + startY - feet.y },
    table_offset: table ? [feet.x - table.x, feet.y - table.y, feet.z - table.z] : null,
    on_own_table: Boolean(table) && feet.x === table.x && feet.y - 1 === table.y && feet.z === table.z,
    table_dropped_headings: stone.headingsNearTable(feet, table),
    staircases: surveyStaircases(world, feet, scan.ctx),
    fan: {
      ms: Math.round(scan.fan.stats.ms * 100) / 100, rays: scan.fan.stats.rays, reads: scan.fan.stats.reads, hits: scan.fan.stats.hits,
      targets: scan.fan.targets.length, targets_any_light: anyLight.fan, nearest: scan.fan.targets.slice(0, 3).map(fanTargetSummary)
    },
    reach: { ms: Math.round(scan.reach_ms * 100) / 100, candidates: scan.reach.length, candidates_any_light: anyLight.reach }
  }
  record.survey_ms = Math.round(Number(process.hrtime.bigint() - started) / 1e4) / 100
  return record
}

async function main () {
  const config = parseArgs(process.argv.slice(2))
  const mineflayer = require('mineflayer')
  const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
  const Move = require('mineflayer-pathfinder/lib/move')
  const stone = stoneHelpers()
  class GoalPickup extends goals.GoalNear {
    // Ends only on nodes from which the vanilla pickup box reaches the item.
    constructor (entity) { super(entity.position.x, entity.position.y, entity.position.z, 1); this.item = entity.position.clone() }
    isEnd (node) { return nodeCollectsItem(node, this.item) }
  }
  const connection = { host: config.host, port: config.port, username: config.username, version: config.version }
  const bot = mineflayer.createBot({ ...connection, auth: 'offline', hideErrors: true, respawn: false })
  bot.loadPlugin(pathfinder)
  let connected = false
  let initialized = false
  let inventoryReceived = false
  let ended = false
  let fatal = null
  let spawn = null
  let ownTable = null
  let explorationIndex = 0
  let placementIndex = 0
  // The most recent bounded recovery and the last action's diagnostic detail;
  // both are reported in the step's final observation and reset per request.
  let recovery = null
  let actionDetail = null
  const searchMemory = new SearchMemory()
  // The stone observation fields of section 3.6 (all but the two inventory
  // counts, which inventoryCounts always reports and the wood state ignores)
  // are on every observation under the stone goal or the survey: time_of_day
  // always, the others null until a wooden pickaxe is first in the
  // inventory. Without either they are absent and no stone scan runs, so a
  // wood observation differs from the bridge of 87673b8 and 409560e only by
  // inventory_counts.stone_materials and stone_pickaxes, both 0 in a wood
  // episode, and no state, reward or action changes. stone-helpers itself is
  // loaded once, at the start of main(), under every goal.
  const stoneFields = config.goal === 'stone_pickaxe' || config.stoneSurvey
  const stoneMemory = new stone.StoneMemory()
  let stoneStarted = false
  // The stage-2 start (stageStartReference), latched at the first observation
  // with the wooden pickaxe held: the hand-off y and the natural floor under it.
  let stageStart = null
  let pickaxeDigs = 0
  // The climb search every stone check shares, built once the spawn is known.
  let stoneClimb = null
  const log = message => process.stderr.write(`[natural-resource bridge] ${message}\n`)
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
  bot.on('spawn', () => { connected = true })
  bot._client.on('window_items', packet => { if (packet.windowId === 0) inventoryReceived = true })
  bot.on('error', error => { fatal = error; log(error.message) })
  bot.on('kicked', reason => { fatal = new Error(`Kicked: ${JSON.stringify(reason)}`); log(fatal.message) })
  bot.on('end', reason => { connected = false; ended = true; log(`Connection ended: ${reason}`) })

  function stop () {
    bot.pathfinder.setGoal(null)
    bot.clearControlStates()
    try { bot.stopDigging() } catch (_) {}
  }
  function check (signal) {
    if (signal.aborted) throw new Error('Request deadline exceeded; this episode is invalid')
    if (fatal) throw fatal
    if (ended) throw new Error('Minecraft connection closed')
  }
  async function until (predicate, timeout, signal, message) {
    const deadline = Date.now() + timeout
    while (!predicate()) {
      check(signal)
      if (Date.now() >= deadline) throw new Error(message)
      await sleep(50)
    }
    check(signal)
  }
  async function waitFor (predicate, timeout, signal) {
    const deadline = Date.now() + timeout
    while (!predicate()) {
      check(signal)
      if (Date.now() >= deadline) return false
      await sleep(50)
    }
    check(signal)
    return true
  }
  function inventory () {
    const result = {}
    for (const item of bot.inventory.items()) result[item.name] = (result[item.name] || 0) + item.count
    return result
  }
  function inBounds (position, margin = 0) { return withinRadius(spawn, position, margin) }
  // Visibility uses the face-sampling check above instead of the library's
  // single ray to a block's centre: in two retained PC failures that ray did
  // not reach a table one block away. Its range limit was not the cause, and
  // the transitions do not record which block stopped it.
  function canSee (block, options) { return canSeeFace(bot.world, eyesAt(bot.entity.position), block, options) }
  function visibleLogs (maxDistance = SEARCH_RADIUS) {
    // Structured loaded-block observations, with a line-of-sight check; no map,
    // seed knowledge, offscreen chunk loading, or privileged server lookup.
    return bot.findBlocks({ matching: block => block.name.endsWith('_log') && !block.name.startsWith('stripped_'), maxDistance, count: 64 })
      .map(position => bot.blockAt(position))
      .filter(block => block && inBounds(block.position, TARGET_MARGIN) && canSee(block))
      .sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
  }
  function reachableLogs () {
    const eyes = bot.entity.position.offset(0, 1.62, 0)
    return visibleLogs(6)
      .filter(block => bot.canDigBlock(block) && block.position.offset(0.5, 0.5, 0.5).distanceTo(eyes) <= 4.5 && !underHitbox(bot.entity.position, block.position))
      // Lowest first: a trunk's bottom log drops its item onto open ground
      // instead of onto the remaining trunk, and keeps the next log in reach.
      .sort((a, b) => a.position.y - b.position.y || a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
  }
  function reachableLog () { return reachableLogs()[0] || null }
  function trunkTargets () {
    // One target per trunk column: its lowest visible log, so the approach goal
    // sits beside the trunk base rather than at a block high in the canopy.
    const lowest = new Map()
    for (const block of visibleLogs()) {
      // The log the bot stands on is never a target: CHOP would refuse it, and
      // a failed approach would block that trunk in the search memory.
      if (underHitbox(bot.entity.position, block.position)) continue
      const key = `${block.position.x},${block.position.z}`
      if (!lowest.has(key) || block.position.y < lowest.get(key).position.y) lowest.set(key, block)
    }
    return [...lowest.values()].sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
  }
  function droppedItemNear (position, maxDistance = 6) {
    const center = position.offset(0.5, 0.5, 0.5)
    return bot.nearestEntity(entity => entity.name === 'item' && entity.position.distanceTo(center) <= maxDistance)
  }
  function tableBlock () {
    const block = ownTable && bot.blockAt(ownTable)
    return block?.name === 'crafting_table' ? block : null
  }
  function tableInReach () {
    const block = tableBlock()
    return block && block.position.offset(0.5, 0.5, 0.5).distanceTo(eyesAt(bot.entity.position)) <= 4.5 && canSee(block) ? block : null
  }
  function supportBlock () {
    // The block carrying the bot: its own column, or a corner column when it
    // stands on the edge of a block and its own column is air.
    const position = bot.entity.position
    const feet = position.floored()
    const own = bot.blockAt(feet.offset(0, -1, 0))
    if (own?.boundingBox === 'block') return own
    for (const [dx, dz] of [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]]) {
      const corner = bot.blockAt(new Vec3(Math.floor(position.x + dx), feet.y - 1, Math.floor(position.z + dz)))
      if (corner?.boundingBox === 'block') return corner
    }
    return null
  }
  function onCanopy () {
    const support = supportBlock()
    return Boolean(support && isLeaves(support.name) && bot.entity.onGround)
  }
  function columnBelow (x, feetY, z, depth = CANOPY_SCAN_DEPTH) {
    return Array.from({ length: depth }, (_, index) => {
      const block = bot.blockAt(new Vec3(x, feetY - 1 - index, z))
      if (!block) return null
      // Collision top inside the cell: carpets, slabs, mud and paths are solid
      // but lower than a full cube, so the real fall onto them is longer.
      const top = Array.isArray(block.shapes) && block.shapes.length ? Math.max(...block.shapes.map(shape => shape[4])) : 1
      return { name: block.name, boundingBox: block.boundingBox, top }
    })
  }
  function standable (position) { return standableIn(bot.world, position) }
  function obstacleBoxes () {
    // Creatures near the bot block placement in vanilla; spectators, dropped
    // items and experience orbs do not.
    const spectators = new Set(Object.values(bot.players).filter(player => player.gamemode === 3 && player.entity).map(player => player.entity))
    return Object.values(bot.entities)
      .filter(entity => entity !== bot.entity && entity.position && !spectators.has(entity) && !['item', 'experience_orb'].includes(entity.name) &&
        Number.isFinite(entity.width) && Number.isFinite(entity.height) && entity.position.distanceTo(bot.entity.position) <= 12)
      .map(entity => ({
        x0: entity.position.x - entity.width / 2, x1: entity.position.x + entity.width / 2, y0: entity.position.y,
        y1: entity.position.y + entity.height, z0: entity.position.z - entity.width / 2, z1: entity.position.z + entity.width / 2
      }))
  }
  function alternativeFooting (feet, tried = new Set()) {
    // A neighbouring leaf block, level or one step down, whose own column has a
    // verified landing. Pathfinder handles these ordinary one-block moves; its
    // step-down move also needs the neighbour clear at the bot's own feet and
    // head, and a neighbour it already failed to reach is not offered again.
    let best = null
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      for (const dy of [0, -1]) {
        const candidate = feet.offset(dx, dy, dz)
        if (tried.has(candidate.toString())) continue
        const support = bot.blockAt(candidate.offset(0, -1, 0))
        if (!support || !isLeaves(support.name) || support.boundingBox !== 'block' || !standable(candidate) || !inBounds(candidate, TARGET_MARGIN)) continue
        if (dy === -1 && !standable(candidate.offset(0, 1, 0))) continue
        const plan = planCanopyDescent({ feetY: candidate.y, health: bot.health, column: columnBelow(candidate.x, candidate.y, candidate.z) })
        if (plan.ok && (!best || plan.fall < best.plan.fall)) best = { position: candidate, plan }
      }
    }
    return best
  }
  function standingCell () {
    // The cell the pathfinder starts from: one higher when the bot stands on a
    // solid block lower than a full cube (path, farmland, mud, slab).
    const position = bot.entity.position
    const cell = position.floored()
    const block = bot.blockAt(cell)
    if (block && position.y - cell.y > 0.001 && bot.entity.onGround && block.boundingBox !== 'empty') cell.y += 1
    return cell
  }
  function enclosed () {
    // Ask the bridge's own movement rules whether any legal move exists from
    // the current cell: a walk, a one-block jump up, or a short drop onto a
    // safe landing. The rules that reject water, lava, bushes, powder snow and
    // long drops for pathfinding apply here too, and they never dig or place.
    if (!bot.entity.onGround) return false
    const cell = standingCell()
    return bot.pathfinder.movements.getNeighbors(new Move(cell.x, cell.y, cell.z, 0, 0)).length === 0
  }
  async function escapeFoliage (signal) {
    // Bounded leaf clearing beside the bot. A descent through a canopy's lowest
    // layers can leave the bot in a one-block shaft walled by leaves, where the
    // pathfinder, which never digs, has no move at all. Clear at most
    // MAX_ESCAPE_LEAVES leaf blocks in one direction that has a floor; never a
    // log, dirt, or stone, and nothing when a non-leaf wall is the obstacle.
    const feet = standingCell()
    const record = { type: 'leaf_escape', blocks_dug: [], direction: null, floor: null, walls: null, escaped: false, items_picked_up: {}, stopped: null }
    // Published at once, like the canopy descent, so a partial escape is still
    // observed when a dig throws, for example after a death mid-dig.
    if (recovery) recovery.escape = record; else recovery = record
    const inventoryBefore = inventory()
    const headings = { '+x': [1, 0], '-x': [-1, 0], '+z': [0, 1], '-z': [0, -1] }
    const describe = block => block ? { name: block.name, boundingBox: block.boundingBox } : null
    const cells = {}
    const descriptors = {}
    for (const [key, [dx, dz]] of Object.entries(headings)) {
      cells[key] = { feet: bot.blockAt(feet.offset(dx, 0, dz)), head: bot.blockAt(feet.offset(dx, 1, dz)), below: bot.blockAt(feet.offset(dx, -1, dz)), lower: bot.blockAt(feet.offset(dx, -2, dz)) }
      descriptors[key] = inBounds(feet.offset(dx, 0, dz), TARGET_MARGIN) ? Object.fromEntries(Object.entries(cells[key]).map(([name, block]) => [name, describe(block)])) : null
    }
    record.walls = descriptors
    const plan = planLeafEscape(descriptors)
    if (!plan.ok) { record.stopped = plan.reason; return record }
    record.direction = plan.direction
    record.floor = plan.floor
    record.exit = { x: feet.x + headings[plan.direction][0], y: feet.y, z: feet.z + headings[plan.direction][1] }
    try {
      for (const block of [cells[plan.direction].feet, cells[plan.direction].head].filter(block => block && isLeaves(block.name))) {
        if (record.blocks_dug.length >= MAX_ESCAPE_LEAVES) { record.stopped = 'block_limit'; break }
        await digBlock(block, signal)
        await waitFor(() => bot.blockAt(block.position)?.boundingBox === 'empty', 1500, signal)
        if (bot.blockAt(block.position)?.boundingBox !== 'empty') { record.stopped = 'dig_unconfirmed'; break }
        record.blocks_dug.push(block.name)
      }
    } catch (error) {
      record.stopped = record.stopped || `error:${error.message}`
      throw error
    } finally {
      record.escaped = !enclosed()
      // Leaves broken by hand roll the vanilla loot table; attribute pickups here.
      for (const [name, count] of Object.entries(inventory())) {
        if (count > (inventoryBefore[name] || 0)) record.items_picked_up[name] = count - (inventoryBefore[name] || 0)
      }
    }
    return record
  }
  function observe (event = 'observe', duration = 0) {
    const inv = inventory()
    const counts = inventoryCounts(inv)
    const biome = bot.blockAt(bot.entity.position)?.biome
    const feet = bot.entity.position.floored()
    const adjacentDrops = [[1, 0], [-1, 0], [0, 1], [0, -1]].map(([dx, dz]) => {
      for (let depth = 1; depth <= 8; depth++) {
        const ground = bot.blockAt(feet.offset(dx, -depth, dz))
        if (!ground) return { dx, dz, ground_block: null, feet_drop: null, loaded: false }
        if (ground.boundingBox === 'block') return { dx, dz, ground_block: ground.name, feet_drop: depth - 1, loaded: true }
      }
      return { dx, dz, ground_block: null, feet_drop: null, loaded: true }
    })
    const support = supportBlock()
    const result = validatePhysicalState({
      position: { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z },
      health: bot.health, food: bot.food, gamemode: bot.game.gameMode,
      inventory: inv, inventory_counts: counts,
      nearby_log: Boolean(reachableLog()), nearby_table: Boolean(tableInReach()), table_placed: Boolean(tableBlock()),
      nearby_visible_logs: visibleLogs().slice(0, 3).map(block => ({ name: block.name, x: block.position.x, y: block.position.y, z: block.position.z })),
      biome: bot.registry.biomes?.[biome?.id ?? biome]?.name || biome?.name || null,
      feet_block: bot.blockAt(feet)?.name || null,
      head_block: bot.blockAt(feet.offset(0, 1, 0))?.name || null,
      ground_block: bot.blockAt(feet.offset(0, -1, 0))?.name || null,
      support_block: support?.name || null,
      on_canopy: onCanopy(),
      adjacent_drops: adjacentDrops,
      table_position: tableBlock() ? { x: ownTable.x, y: ownTable.y, z: ownTable.z } : null,
      held_item: bot.heldItem?.name || null, success: goalReached(config.goal, counts, bot.heldItem?.name), death: bot.health <= 0,
      recovery, action_detail: actionDetail,
      event, action_duration_seconds: duration, backend: BACKEND, game_version: bot.version
    })
    if (!inBounds(result.position)) throw new Error('Bot left the 64-block starting radius; episode invalid')
    if (stoneFields) Object.assign(result, observeStone(counts))
    return result
  }
  // The context of every stone check: the observation's scan, and from step
  // S3 MINE_STONE and DIG_DOWN, so nearby_stone is MINE_STONE's precondition.
  // pickaxeType is the wooden pickaxe while one is in the inventory (MINE_STONE
  // equips it), otherwise null, and pickaxe blocks then refuse as wrong_tool.
  function stoneScene () {
    stoneClimb = stoneClimb || stone.climbSearch(fake => stone.strictMovements(fake, Movements, spawn), Move)
    return {
      botPosition: bot.entity.position, spawn, table: ownTable, entities: bot.entities, startY: stageStart?.y ?? null, climb: stoneClimb,
      // Step S3: DIG_DOWN's live staircase record { heading, start, landings }, shared with MINE_STONE.
      stair: null,
      pickaxeType: bot.inventory.items().some(item => item.name === 'wooden_pickaxe') ? stone.WOODEN_PICKAXE : null
    }
  }
  function observeStone (counts) {
    const fields = {
      nearby_stone: null, stone_visible: null, stone_targets: null, stone_targets_withheld: null, stone_reach_candidates: null, pickaxe_uses_left: null,
      pickaxe_digs: null, hostile_nearby: null, time_of_day: null, hand_off_y: null, stage_start_y: null, depth_below_stage_start: null
    }
    if (config.stoneSurvey) fields.diagnostic_only = null
    // On every observation (section 4.10): it reads no blocks.
    fields.time_of_day = Number.isFinite(bot.time?.timeOfDay) ? bot.time.timeOfDay : null
    // Latched, so a pickaxe that breaks later does not remove the fields.
    stoneStarted = stoneStarted || counts.pickaxes > 0
    if (!stoneStarted) return fields
    const feet = standingCell()
    try {
      // Latched once and never re-derived at a later position.
      if (stageStart === null && bot.heldItem?.name === 'wooden_pickaxe') stageStart = stageStartReference(bot.world, feet, ownTable)
      const scan = stoneScan(bot.world, stoneScene(), { memory: stoneMemory })
      Object.assign(fields, scan.fields, {
        pickaxe_uses_left: pickaxeUsesLeft(bot.inventory.items(), bot.heldItem), pickaxe_digs: pickaxeDigs,
        hand_off_y: stageStart?.hand_off_y ?? null, stage_start_y: stageStart?.y ?? null,
        depth_below_stage_start: stageStart === null ? null : stageStart.y - feet.y
      })
      if (config.stoneSurvey && stageStart !== null) fields.diagnostic_only = stoneSurveyRecord(bot.world, feet, scan, { memory: stoneMemory, stageStart })
    } catch (error) {
      // Under the wood goal these fields are survey diagnostics only: a
      // failure is recorded and never ends a wood episode.
      if (config.goal === 'stone_pickaxe') throw error
      log(`Stone survey failed: ${error.message}`)
      fields.diagnostic_only = { error: error.message }
    }
    return fields
  }
  async function initialize (signal) {
    if (initialized) return
    await until(() => connected && bot.entity && bot.inventory && inventoryReceived && Number.isFinite(bot.health) && Number.isFinite(bot.food), 30000, signal, 'No complete Minecraft spawn received')
    await bot.waitForChunksToLoad()
    await sleep(1000)
    check(signal)
    spawn = bot.entity.position.clone()
    if (Object.keys(inventory()).length) throw new Error('Fresh episode requires an empty bot inventory; start a new world')
    if (bot.health <= 0) throw new Error('Bot died before initialization; start a new world')
    bot.pathfinder.setMovements(safeMovements(bot, Movements, spawn))
    bot.pathfinder.thinkTimeout = 2500
    bot.pathfinder.tickTimeout = 25
    bot.pathfinder.searchRadius = 64
    observe('spawn')
    initialized = true
  }
  async function navigate (goal, timeout, signal) {
    let timer
    try {
      const reached = await Promise.race([
        bot.pathfinder.goto(goal).then(() => true, error => { log(`Navigation: ${error.message}`); return false }),
        new Promise(resolve => { timer = setTimeout(() => resolve(false), timeout) })
      ])
      check(signal)
      // The pinned goto implementation can resolve on an empty unsuccessful
      // path. Verify actual coordinates against the goal, not its return value.
      return reached && goal.isEnd(bot.entity.position.floored())
    } finally {
      clearTimeout(timer)
      bot.pathfinder.setGoal(null)
      bot.clearControlStates()
    }
  }
  async function digBlock (block, signal) {
    // A dead bot never finishes mineflayer's pre-dig look, and the digging
    // plugin's own death handler only cancels a dig already in progress, so
    // bound every dig by its expected duration and by a death event instead of
    // leaving it to the 44-second request deadline.
    if (bot.health <= 0) throw new Error('Bot died before digging')
    const withPickaxe = bot.heldItem?.name === 'wooden_pickaxe'
    let timer
    let onDeath
    const died = new Promise((_, reject) => { onDeath = () => reject(new Error('Bot died while digging')); bot.once('death', onDeath) })
    try {
      await Promise.race([
        bot.dig(block),
        died,
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Dig did not complete in time')), bot.digTime(block) + 3000) })
      ])
      if (withPickaxe) pickaxeDigs++
    } finally {
      clearTimeout(timer)
      bot.removeListener('death', onDeath)
      if (bot.targetDigBlock) { try { bot.stopDigging() } catch (_) {} }
    }
    check(signal)
  }
  async function descendCanopy (signal) {
    // Bounded scripted recovery for a bot standing on tree leaves: clear only
    // the leaf block under the feet when the column below has a verified
    // landing, and record every cleared block, fall, and health change. No
    // other block is dug, nothing is placed, and nothing is supplied.
    const record = { type: 'canopy_descent', layers_cleared: 0, blocks_dug: [], falls: [], moves: 0, damage: 0, items_picked_up: {}, start_y: bot.entity.position.y, end_y: null, descended: false, stopped: null, refused: null }
    // Published at once, so a partial descent is still observed when a dig or
    // move throws, for example after a death mid-dig.
    recovery = record
    const inventoryBefore = inventory()
    const tried = new Set()
    const deadline = Date.now() + CANOPY_BUDGET_MS
    const offCentre = support => Math.abs(bot.entity.position.x - (support.position.x + 0.5)) > 0.2 || Math.abs(bot.entity.position.z - (support.position.z + 0.5)) > 0.2
    try {
      for (let iteration = 0; iteration < MAX_CANOPY_LAYERS * 2; iteration++) {
        check(signal)
        if (record.layers_cleared >= MAX_CANOPY_LAYERS) { record.stopped = 'layer_limit'; break }
        if (Date.now() >= deadline) { record.stopped = 'time_budget'; break }
        if (bot.health <= 0) { record.stopped = 'died'; break }
        const support = supportBlock()
        if (!support || !isLeaves(support.name)) break
        const feet = bot.entity.position.floored()
        if (support.position.x !== feet.x || support.position.z !== feet.z || offCentre(support)) {
          // Stand centred on the supporting leaf block so clearing it drops the
          // bot straight down. The hitbox is 0.6 wide: an offset above 0.2
          // overlaps a neighbouring column that would hold the bot up. The
          // pathfinder recentres the bot when it reaches a goal block.
          record.moves++
          const reached = await navigate(new goals.GoalBlock(support.position.x, feet.y, support.position.z), 5000, signal)
          if (!reached || offCentre(support)) { record.stopped = 'could_not_centre_on_support'; break }
          continue
        }
        const plan = planCanopyDescent({ feetY: feet.y, health: bot.health, column: columnBelow(feet.x, feet.y, feet.z) })
        if (!plan.ok) {
          const alternative = alternativeFooting(feet, tried)
          if (!alternative) { record.stopped = tried.size ? 'alternative_footing_unreachable' : plan.reason; record.refused = plan; break }
          record.moves++
          const p = alternative.position
          // A neighbour the pathfinder cannot reach is skipped on the next try
          // instead of ending the descent for every later SEARCH.
          if (!await navigate(new goals.GoalBlock(p.x, p.y, p.z), 6000, signal)) tried.add(p.toString())
          continue
        }
        const healthBefore = bot.health
        const yBefore = bot.entity.position.y
        await digBlock(support, signal)
        const landed = await waitFor(() => bot.entity.onGround && bot.entity.position.y < yBefore - 0.5, 5000, signal)
        await sleep(300) // let the server's health and block updates arrive before measuring
        check(signal)
        // Count a layer only when the server shows the block gone.
        if (bot.blockAt(support.position)?.boundingBox !== 'empty') { record.stopped = 'dig_unconfirmed'; break }
        record.layers_cleared++
        record.blocks_dug.push(support.name)
        const fall = {
          planned_fall: plan.fall, expected_damage: plan.expected_damage, landing_block: plan.landing_block,
          actual_fall: Math.round((yBefore - bot.entity.position.y) * 100) / 100,
          damage: Math.max(0, healthBefore - bot.health), landed
        }
        record.falls.push(fall)
        record.damage += fall.damage
        if (!landed) { record.stopped = 'landing_unconfirmed'; break }
        if (bot.health <= 0) { record.stopped = 'died'; break }
      }
    } catch (error) {
      record.stopped = record.stopped || `error:${error.message}`
      throw error
    } finally {
      record.end_y = bot.entity.position.y
      record.descended = !onCanopy()
      // Leaves broken by hand roll the vanilla loot table (sticks, saplings,
      // apples); anything collected during the descent is attributed here.
      for (const [name, count] of Object.entries(inventory())) {
        if (count > (inventoryBefore[name] || 0)) record.items_picked_up[name] = count - (inventoryBefore[name] || 0)
      }
    }
    return record
  }
  async function search (signal) {
    const started = Date.now()
    const remaining = () => SEARCH_BUDGET_MS - (Date.now() - started)
    const notes = []
    if (onCanopy()) {
      const descent = await descendCanopy(signal)
      recovery = descent
      const cleared = `cleared_${descent.layers_cleared}_leaf_layer${descent.layers_cleared === 1 ? '' : 's'}`
      if (descent.descended) notes.push(cleared, 'descended_canopy')
      else if (descent.layers_cleared) return [cleared, `still_on_canopy:${descent.stopped || 'unknown'}`].join(';')
      // Refused, or unable to reposition: keep the ordinary search so the bot
      // can roam the canopy towards a lower edge instead of repeating the same
      // refusal every decision; the next SEARCH re-evaluates the descent.
      else notes.push(`canopy_descent_refused:${descent.stopped || 'unknown'}`)
    }
    if (reachableLog()) { searchMemory.reached(); return [...notes, 'log_already_reachable'].join(';') }
    if (enclosed()) {
      const escape = await escapeFoliage(signal) // publishes its own record
      notes.push(escapeNote(escape))
    }
    const candidate = searchMemory.next(trunkTargets())
    if (candidate) {
      const p = candidate.position
      const reached = await navigate(new goals.GoalNear(p.x, p.y, p.z, 2), Math.max(4000, Math.min(18000, remaining())), signal)
      if (reached && reachableLog()) { searchMemory.reached(); return [...notes, 'log_reached'].join(';') }
      searchMemory.failed(candidate)
      return [...notes, 'log_path_blocked'].join(';')
    }
    // Exploration tries several targets within one decision instead of spending
    // a whole decision on each heading the pathfinder rejects at once. Long
    // targets are spawn-relative rings that cover every heading before the
    // radius grows; a short move from the current position only follows a
    // long target that could not be reached.
    for (let attempt = 0; attempt < 4; attempt++) {
      if (remaining() < 4000) { notes.push('exploration_time_budget'); break }
      const { dx, dz, radius } = explorationTarget(explorationIndex++)
      const target = { x: spawn.x + dx * radius, z: spawn.z + dz * radius }
      if (!inBounds(target, TARGET_MARGIN)) { notes.push('exploration_boundary'); continue }
      if (await navigate(new goals.GoalNearXZ(target.x, target.z, 2), Math.min(18000, remaining()), signal)) return [...notes, 'explored'].join(';')
      const local = { x: bot.entity.position.x + dx * 4, z: bot.entity.position.z + dz * 4 }
      if (remaining() >= 4000 && inBounds(local, TARGET_MARGIN) && await navigate(new goals.GoalNearXZ(local.x, local.z, 1), Math.min(6000, remaining()), signal)) return [...notes, 'explored_locally'].join(';')
    }
    return [...notes, 'exploration_blocked'].join(';')
  }
  async function chop (signal) {
    // Dig a log the bot can already see and reach from where it stands. The
    // observation's nearby_log comes from the same list, so the state and the
    // action agree. CHOP never walks first: an approach move can put a trunk's
    // bottom log behind the log above it, and a frozen policy whose state does
    // not change would then repeat the same failing decision until truncation.
    const candidates = reachableLogs().slice(0, 3)
    if (!candidates.length) return 'no_reachable_log'
    const detail = { type: 'chop', candidates: candidates.map(block => [block.position.x, block.position.y, block.position.z]), target: null, item_seen: false, item_position: null, pickup_moves: 0, collected: false }
    actionDetail = detail
    const before = inventoryCounts(inventory()).logs
    let target = null
    for (const candidate of candidates) {
      const block = bot.blockAt(candidate.position)
      if (block?.name.endsWith('_log') && canSee(block) && bot.canDigBlock(block) && !underHitbox(bot.entity.position, block.position)) { target = block; break }
    }
    if (!target) return 'log_unreachable'
    const p = target.position.clone()
    detail.target = [p.x, p.y, p.z]
    await digBlock(target, signal)
    const collected = () => inventoryCounts(inventory()).logs > before
    // Walk to the dropped item itself: a high trunk block's drop lands wherever
    // gravity takes it, not at the removed block. Fall back to the block's
    // column at the current standing level when no item entity is visible.
    const deadline = Date.now() + 9000
    await waitFor(() => collected() || Boolean(droppedItemNear(p)), 1000, signal)
    for (let attempt = 0; attempt < 2 && !collected() && Date.now() < deadline && bot.health > 0; attempt++) {
      const drop = droppedItemNear(p)
      if (drop) { detail.item_seen = true; detail.item_position = [drop.position.x, drop.position.y, drop.position.z].map(v => Math.round(v * 100) / 100) }
      // An item that rolled towards the edge of the search area is left where it is.
      // The walk is bounded by the pathfinder's STEP_MARGIN, which also bounds
      // every drop of a log CHOP accepted, so only a drop that rolled farther is left.
      if (drop && !inBounds(drop.position, STEP_MARGIN)) { detail.item_out_of_bounds = true; break }
      detail.pickup_moves++
      const goal = drop ? new GoalPickup(drop) : new goals.GoalNear(p.x, Math.floor(bot.entity.position.y), p.z, 1)
      await navigate(goal, Math.max(1000, Math.min(6000, deadline - Date.now())), signal)
      await waitFor(collected, 2000, signal)
    }
    detail.collected = collected()
    return detail.collected ? 'log_collected' : 'log_broken_without_pickup'
  }
  async function craft (action, signal) {
    const before = inventory()
    const targets = action === 2
      ? Object.keys(before).filter(name => name.endsWith('_log')).map(name => name.replace(/^stripped_/, '').replace(/_log$/, '_planks'))
      : [{ 3: 'stick', 4: 'crafting_table', 7: 'wooden_pickaxe' }[action]]
    const table = action === 7 ? tableInReach() : null
    if (action === 7 && !table) return 'not_at_own_table'
    for (const name of targets) {
      const item = bot.registry.itemsByName[name]
      if (!item) continue
      const recipe = craftingRecipe(bot, item, table)
      if (!recipe) continue
      // Pinned 4.39.0 uses optimistic 2x2 updates. Ask the server to resync the
      // inventory even after a failed craft, so it cannot leave a phantom tool.
      try { await bot.craft(recipe, 1, table) } finally { await syncInventory(signal) }
      if ((inventory()[name] || 0) <= (before[name] || 0)) return 'craft_not_confirmed'
      if (action === 7) {
        try {
          await bot.equip(bot.inventory.items().find(item => item.name === 'wooden_pickaxe'), 'hand')
          await syncInventory(signal)
          await sleep(250)
          check(signal)
          if (bot.heldItem?.name !== 'wooden_pickaxe') throw new Error('Equipped item was not the crafted pickaxe')
        } catch (error) { throw new ObservationError(`Crafted pickaxe could not be equipped: ${error.message}`) }
      }
      return `crafted_${name}`
    }
    return 'missing_materials'
  }
  async function syncInventory (signal) {
    check(signal)
    if (typeof bot._syncWindow !== 'function') throw new ObservationError('Pinned inventory sync API is missing')
    try { await bot._syncWindow(bot.inventory) } catch (error) { throw new ObservationError(`Inventory sync failed: ${error.message}`) }
    check(signal)
  }
  async function placeTable (signal) {
    if (tableBlock()) return 'table_already_placed'
    const item = bot.inventory.items().find(item => item.name === 'crafting_table')
    if (!item) return 'missing_table'
    let excludedColumns = []
    const placementTargets = (radius, withinReach) => placementCandidates(bot.world, bot.entity.position, spawn, { radius, withinReach, obstacles: obstacleBoxes(), excludedColumns })
    let note = null
    const label = event => note ? `${note};${event}` : event
    let target = placementTargets(3, true)[0]
    if (!target && enclosed()) {
      // Walled in with nothing to place on: the same bounded, recorded leaf
      // clearing SEARCH uses, named in this step's event (unless a later part
      // of the action fails), then look again. The exit it opened is never
      // filled with the table.
      const escape = await escapeFoliage(signal)
      note = escapeNote(escape)
      if (escape.exit && escape.blocks_dug.length) excludedColumns = [escape.exit]
      target = placementTargets(3, true)[0]
    }
    if (!target) {
      const candidates = placementTargets(8, false).slice(0, 12)
      const destination = candidates.length ? candidates[placementIndex++ % candidates.length] : null
      if (!destination) return label('no_placement_surface')
      const p = destination.position
      await navigate(new goals.GoalNear(p.x, p.y, p.z, 1.5), 12000, signal)
      target = placementTargets(3, true)[0]
    }
    if (!target) return label('no_placement_surface')
    await bot.equip(item, 'hand')
    await bot.placeBlock(target.support, new Vec3(0, 1, 0))
    await until(() => bot.blockAt(target.position)?.name === 'crafting_table', 3000, signal, 'Table placement was not observed')
    await syncInventory(signal)
    check(signal)
    ownTable = target.position.clone()
    return label('table_placed')
  }
  async function step (request, signal) {
    const action = runnableAction(request.action, config.goal)
    recovery = null
    actionDetail = null
    const before = observe()
    if (before.death) return { ...before, event: 'died_before_action', action_duration_seconds: 0 }
    if (before.success) return { ...before, event: 'already_completed', action_duration_seconds: 0 }
    const started = Date.now()
    let event
    try {
      if (action === 0) event = await search(signal)
      else if (action === 1) event = await chop(signal)
      else if ([2, 3, 4, 7].includes(action)) event = await craft(action, signal)
      else if (action === 5) event = await placeTable(signal)
      else {
        const target = tableBlock()
        if (!target) event = 'no_own_table'
        else {
          // Stand where a face of the table is visible: near it first, then on
          // each standable neighbour cell (orthogonal first) until it is in reach.
          const p = target.position
          const deadline = Date.now() + 18000
          let reached = tableInReach() ? true : await navigate(new goals.GoalNear(p.x, p.y, p.z, 2), 12000, signal)
          for (const cell of approachCells(bot.world, p, spawn)) {
            if (tableInReach() || Date.now() >= deadline) break
            reached = await navigate(new goals.GoalBlock(cell.x, cell.y, cell.z), Math.max(1500, Math.min(6000, deadline - Date.now())), signal) || reached
          }
          event = tableInReach() ? 'table_reached' : (reached ? 'table_not_visible' : 'table_path_blocked')
        }
      }
    } catch (error) {
      check(signal)
      if (error instanceof ObservationError) throw error
      // Ordinary skill execution failures are observed transitions, never
      // invented success. Connection/invalid-state failures propagate below.
      log(`${ACTIONS[action]} failed: ${error.message}`)
      event = `action_failed: ${error.message}`
    } finally { stop() }
    check(signal)
    return observe(event, (Date.now() - started) / 1000)
  }
  async function dispatch (request, signal) {
    if (!request || typeof request !== 'object' || Array.isArray(request)) throw new Error('Request must be a JSON object')
    if (request.op === 'close') return { closed: true }
    await initialize(signal)
    check(signal)
    if (request.op === 'status') return {
      connected, initialized, spawn: { x: spawn.x, y: spawn.y, z: spawn.z }, gamemode: bot.game.gameMode,
      game_version: bot.version, backend: BACKEND,
      players: Object.fromEntries(Object.entries(bot.players)
        .filter(([, player]) => Number.isInteger(player.gamemode) && player.gamemode >= 0 && player.gamemode <= 3)
        .map(([name, player]) => [name, { gamemode: player.gamemode }]))
    }
    // A standalone observation never carries a previous step's recovery.
    if (request.op === 'observe') { recovery = null; actionDetail = null; return observe() }
    if (request.op === 'step') return step(request, signal)
    throw new Error(`Unknown operation: ${request.op}; this bridge has no reset or command operation`)
  }
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
  let queue = Promise.resolve()
  let closing = false
  input.on('line', line => {
    queue = queue.then(async () => {
      if (closing) return
      let request
      let timer
      const controller = new AbortController()
      try {
        request = JSON.parse(line)
        const timeout = new Promise((resolve, reject) => {
          timer = setTimeout(() => { controller.abort(); stop(); reject(new Error('Minecraft resource request exceeded 44 seconds; episode invalid')) }, 44000)
        })
        const result = await Promise.race([dispatch(request, controller.signal), timeout])
        process.stdout.write(JSON.stringify({ id: request.id ?? null, ok: true, result }) + '\n')
        if (request.op === 'close') closing = true
      } catch (error) {
        process.stdout.write(JSON.stringify({ id: request?.id ?? null, ok: false, error: error.message }) + '\n')
        // Stop on every protocol/observation failure; no half-finished action
        // can continue while another request is processed.
        closing = true
      } finally {
        clearTimeout(timer)
        stop()
        if (closing) { input.close(); bot.quit(); setTimeout(() => process.exit(0), 150) }
      }
    }).catch(error => { log(error.stack); process.exitCode = 1 })
  })
  input.on('close', () => {
    queue.finally(() => { closing = true; stop(); bot.quit(); setTimeout(() => process.exit(0), 150) })
  })
  process.on('SIGINT', () => { stop(); bot.quit(); process.exit(0) })
}

module.exports = {
  parseArgs, validateAction, inventoryCounts, validatePhysicalState, craftingRecipe, safeMovements, SearchMemory,
  replaceableForTable, isLeaves, fallDamage, descentAcceptable, planCanopyDescent, nodeCollectsItem, planLeafEscape,
  explorationTarget, underHitbox, withinRadius, eyesAt, escapeNote, visibilitySamples, canSeeFace, hitboxOverlapsCell,
  placementRank, placementCandidates, standableIn, approachCells, interactiveSupport, cellIntersectsBox, stepExclusion,
  passableCell, walkableFloor, ACTIONS, BACKEND, FALL_HAZARDS, MAX_CANOPY_LAYERS, CANOPY_SCAN_DEPTH, PICKUP_REACH,
  MAX_ESCAPE_LEAVES, EXPLORATION_RINGS, EXPLORATION_DIRECTIONS, MAX_RADIUS, TARGET_MARGIN, STEP_MARGIN, EYE_HEIGHT,
  APPROACH_OFFSETS, collisionTop, HITBOX_HALF_WIDTH,
  WOOD_ACTIONS, GOALS, STONE_LIT_ONLY, SURVEY_STAIR_STEPS, SURVEY_DEPTH, actionsFor, runnableAction, goalReached, pickaxeUsesLeft,
  fanTargetSummary, stoneScan, stageStartReference, depthToStone, stairStepSummary, surveyStaircases, stoneSurveyRecord
}
if (require.main === module) main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exit(1) })
