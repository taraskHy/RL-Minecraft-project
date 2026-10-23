'use strict'

// Natural-world executor for eight learned macro-actions. This file never sends
// chat commands, creates resources, resets a world, or chooses the next action.
const readline = require('node:readline')

const ACTIONS = ['SEARCH', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE', 'PLACE_TABLE', 'APPROACH_TABLE', 'CRAFT_PICKAXE']
const BACKEND = 'minecraft-java-natural-resources'
const MAX_RADIUS = 64
const SEARCH_RADIUS = 32
// One request may take 44 seconds; SEARCH keeps a margin for its two observations.
const SEARCH_BUDGET_MS = 36000
const CANOPY_BUDGET_MS = 20000
const MAX_CANOPY_LAYERS = 6
const CANOPY_SCAN_DEPTH = 12
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

function parseArgs (args) {
  const config = { host: '127.0.0.1', port: 25567, username: 'RLResourceBot', version: '1.21.4' }
  const fields = { '--host': 'host', '--port': 'port', '--username': 'username', '--version': 'version' }
  for (let i = 0; i < args.length; i++) {
    if (!fields[args[i]] || args[i + 1] === undefined) throw new Error(`Unknown or incomplete option: ${args[i]}`)
    config[fields[args[i]]] = args[++i]
  }
  config.port = Number(config.port)
  if (!['127.0.0.1', 'localhost', '::1'].includes(config.host)) throw new Error('Natural-resource bridge only connects to localhost')
  if (!Number.isInteger(config.port) || config.port < 1 || config.port > 65535) throw new Error('Invalid port')
  if (!/^[A-Za-z0-9_]{1,16}$/.test(config.username)) throw new Error('Invalid bot username')
  if (config.version !== '1.21.4') throw new Error('Natural-resource bridge is pinned to Minecraft Java 1.21.4')
  return config
}

function validateAction (action) {
  if (!Number.isInteger(action) || action < 0 || action >= ACTIONS.length) throw new Error('action must be an integer from 0 through 7')
  return action
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
  return counts
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
  movements.exclusionAreasStep.push(block => Math.hypot(block.position.x - spawn.x, block.position.z - spawn.z) > MAX_RADIUS ? 100 : 0)
  movements.exclusionAreasPlace.push(() => 100)
  return movements
}

async function main () {
  const config = parseArgs(process.argv.slice(2))
  const mineflayer = require('mineflayer')
  const { Vec3 } = require('vec3')
  const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
  class GoalPickup extends goals.GoalNear {
    // Ends only on nodes from which the vanilla pickup box reaches the item.
    constructor (entity) { super(entity.position.x, entity.position.y, entity.position.z, 1); this.item = entity.position.clone() }
    isEnd (node) { return nodeCollectsItem(node, this.item) }
  }
  const bot = mineflayer.createBot({ ...config, auth: 'offline', hideErrors: true, respawn: false })
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
  // The most recent bounded recovery; reported in the step's final observation.
  let recovery = null
  const searchMemory = new SearchMemory()
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
  function inBounds (position) { return Math.hypot(position.x - spawn.x, position.z - spawn.z) <= MAX_RADIUS }
  function visibleLogs (maxDistance = SEARCH_RADIUS) {
    // Structured loaded-block observations, with a line-of-sight check; no map,
    // seed knowledge, offscreen chunk loading, or privileged server lookup.
    return bot.findBlocks({ matching: block => block.name.endsWith('_log') && !block.name.startsWith('stripped_'), maxDistance, count: 64 })
      .map(position => bot.blockAt(position))
      .filter(block => block && inBounds(block.position) && bot.canSeeBlock(block))
      .sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
  }
  function reachableLogs () {
    const eyes = bot.entity.position.offset(0, 1.62, 0)
    return visibleLogs(6)
      .filter(block => bot.canDigBlock(block) && block.position.offset(0.5, 0.5, 0.5).distanceTo(eyes) <= 4.5)
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
    return block && bot.canSeeBlock(block) && block.position.offset(0.5, 0.5, 0.5).distanceTo(bot.entity.position.offset(0, 1.62, 0)) <= 4.5 ? block : null
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
  function standable (position) {
    const feetBlock = bot.blockAt(position)
    const headBlock = bot.blockAt(position.offset(0, 1, 0))
    return Boolean(feetBlock && headBlock && feetBlock.boundingBox === 'empty' && headBlock.boundingBox === 'empty' &&
      !FALL_HAZARDS.includes(feetBlock.name) && !FALL_HAZARDS.includes(headBlock.name))
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
        if (!support || !isLeaves(support.name) || support.boundingBox !== 'block' || !standable(candidate) || !inBounds(candidate)) continue
        if (dy === -1 && !standable(candidate.offset(0, 1, 0))) continue
        const plan = planCanopyDescent({ feetY: candidate.y, health: bot.health, column: columnBelow(candidate.x, candidate.y, candidate.z) })
        if (plan.ok && (!best || plan.fall < best.plan.fall)) best = { position: candidate, plan }
      }
    }
    return best
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
      held_item: bot.heldItem?.name || null, success: counts.pickaxes > 0, death: bot.health <= 0,
      recovery,
      event, action_duration_seconds: duration, backend: BACKEND, game_version: bot.version
    })
    if (!inBounds(result.position)) throw new Error('Bot left the 64-block starting radius; episode invalid')
    return result
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
    let timer
    let onDeath
    const died = new Promise((_, reject) => { onDeath = () => reject(new Error('Bot died while digging')); bot.once('death', onDeath) })
    try {
      await Promise.race([
        bot.dig(block),
        died,
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Dig did not complete in time')), bot.digTime(block) + 3000) })
      ])
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
    const candidate = searchMemory.next(trunkTargets())
    if (candidate) {
      const p = candidate.position
      const reached = await navigate(new goals.GoalNear(p.x, p.y, p.z, 2), Math.max(4000, Math.min(18000, remaining())), signal)
      if (reached && reachableLog()) { searchMemory.reached(); return [...notes, 'log_reached'].join(';') }
      searchMemory.failed(candidate)
      return [...notes, 'log_path_blocked'].join(';')
    }
    // Local exploration tries several headings within one decision, instead of
    // spending a whole decision on each heading the pathfinder rejects at once.
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    for (let attempt = 0; attempt < 4; attempt++) {
      if (remaining() < 4000) { notes.push('exploration_time_budget'); break }
      const index = explorationIndex++
      const [dx, dz] = directions[index % directions.length]
      const local = index % 2 === 0
      const radius = local ? 4 : 8 + 8 * (Math.floor(index / directions.length) % 4)
      const origin = local ? bot.entity.position : spawn
      const target = { x: origin.x + dx * radius, z: origin.z + dz * radius }
      if (!inBounds(target)) { notes.push('exploration_boundary'); continue }
      const reached = await navigate(new goals.GoalNearXZ(target.x, target.z, local ? 1 : 2), Math.min(18000, remaining()), signal)
      if (reached) return [...notes, 'explored'].join(';')
    }
    return [...notes, 'exploration_blocked'].join(';')
  }
  async function chop (signal) {
    let target = reachableLog()
    if (!target) return 'no_reachable_log'
    const p = target.position.clone()
    const before = inventoryCounts(inventory()).logs
    if (p.distanceTo(bot.entity.position) > 2) await navigate(new goals.GoalNear(p.x, p.y, p.z, 1.5), 10000, signal)
    if (bot.health <= 0) return 'died_during_approach'
    target = bot.blockAt(p)
    if (!target?.name.endsWith('_log') || !bot.canSeeBlock(target) || !bot.canDigBlock(target)) return 'log_unreachable'
    await digBlock(target, signal)
    const collected = () => inventoryCounts(inventory()).logs > before
    // Walk to the dropped item itself: a high trunk block's drop lands wherever
    // gravity takes it, not at the removed block. Fall back to the block's
    // column at the current standing level when no item entity is visible.
    const deadline = Date.now() + 9000
    await waitFor(() => collected() || Boolean(droppedItemNear(p)), 1000, signal)
    for (let attempt = 0; attempt < 2 && !collected() && Date.now() < deadline && bot.health > 0; attempt++) {
      const drop = droppedItemNear(p)
      const goal = drop ? new GoalPickup(drop) : new goals.GoalNear(p.x, Math.floor(bot.entity.position.y), p.z, 1)
      await navigate(goal, Math.max(1000, Math.min(6000, deadline - Date.now())), signal)
      await waitFor(collected, 2000, signal)
    }
    return collected() ? 'log_collected' : 'log_broken_without_pickup'
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
    function placementTargets (radius, withinReach) {
      const base = bot.entity.position.floored()
      const candidates = []
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dz = -radius; dz <= radius; dz++) {
          for (let dy = -3; dy <= 3; dy++) {
            const position = base.offset(dx, dy, dz)
            if (Math.hypot(position.x + 0.5 - bot.entity.position.x, position.z + 0.5 - bot.entity.position.z) < 1.3 || !inBounds(position)) continue
            const empty = bot.blockAt(position)
            const support = bot.blockAt(position.offset(0, -1, 0))
            if (!replaceableForTable(empty?.name) || support?.boundingBox !== 'block') continue
            if (['crafting_table', 'magma_block', 'cactus', 'campfire', 'soul_campfire'].includes(support.name)) continue
            if (withinReach && (!bot.canSeeBlock(support) || support.position.offset(0.5, 1, 0.5).distanceTo(bot.entity.position.offset(0, 1.62, 0)) > 4.3)) continue
            candidates.push({ position, support })
          }
        }
      }
      return candidates.sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
    }
    let target = placementTargets(3, true)[0]
    if (!target) {
      const candidates = placementTargets(8, false).slice(0, 12)
      const destination = candidates.length ? candidates[placementIndex++ % candidates.length] : null
      if (!destination) return 'no_placement_surface'
      const p = destination.position
      await navigate(new goals.GoalNear(p.x, p.y, p.z, 1.5), 12000, signal)
      target = placementTargets(3, true)[0]
    }
    if (!target) return 'no_placement_surface'
    await bot.equip(item, 'hand')
    await bot.placeBlock(target.support, new Vec3(0, 1, 0))
    await until(() => bot.blockAt(target.position)?.name === 'crafting_table', 3000, signal, 'Table placement was not observed')
    await syncInventory(signal)
    check(signal)
    ownTable = target.position.clone()
    return 'table_placed'
  }
  async function step (request, signal) {
    const action = validateAction(request.action)
    recovery = null
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
          const p = target.position
          const reached = await navigate(new goals.GoalNear(p.x, p.y, p.z, 2), 18000, signal)
          event = reached && tableInReach() ? 'table_reached' : 'table_path_blocked'
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
    if (request.op === 'observe') { recovery = null; return observe() }
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
  replaceableForTable, isLeaves, fallDamage, descentAcceptable, planCanopyDescent, nodeCollectsItem, ACTIONS, BACKEND,
  FALL_HAZARDS, MAX_CANOPY_LAYERS, CANOPY_SCAN_DEPTH, PICKUP_REACH
}
if (require.main === module) main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exit(1) })
