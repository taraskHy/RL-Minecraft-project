'use strict'

// Natural-world executor for eight learned macro-actions. This file never sends
// chat commands, creates resources, resets a world, or chooses the next action.
const readline = require('node:readline')

const ACTIONS = ['SEARCH', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_TABLE', 'PLACE_TABLE', 'APPROACH_TABLE', 'CRAFT_PICKAXE']
const BACKEND = 'minecraft-java-natural-resources'
const MAX_RADIUS = 64
const SEARCH_RADIUS = 32
class ObservationError extends Error {}

class SearchMemory {
  constructor () { this.attempt = 0; this.blockedUntil = new Map(); this.consecutiveBlocked = 0 }
  key (block) { return `${block.position.x},${block.position.y},${block.position.z}` }
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
  // maximum three-block feet-height descent, not a four-block fall.
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
  function reachableLog () {
    return visibleLogs(6).find(block => bot.canDigBlock(block) && block.position.offset(0.5, 0.5, 0.5).distanceTo(bot.entity.position.offset(0, 1.62, 0)) <= 4.5) || null
  }
  function tableBlock () {
    const block = ownTable && bot.blockAt(ownTable)
    return block?.name === 'crafting_table' ? block : null
  }
  function tableInReach () {
    const block = tableBlock()
    return block && bot.canSeeBlock(block) && block.position.offset(0.5, 0.5, 0.5).distanceTo(bot.entity.position.offset(0, 1.62, 0)) <= 4.5 ? block : null
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
      adjacent_drops: adjacentDrops,
      table_position: tableBlock() ? { x: ownTable.x, y: ownTable.y, z: ownTable.z } : null,
      held_item: bot.heldItem?.name || null, success: counts.pickaxes > 0, death: bot.health <= 0,
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
  async function search (signal) {
    if (reachableLog()) { searchMemory.reached(); return 'log_already_reachable' }
    const candidate = searchMemory.next(visibleLogs())
    if (candidate) {
      const p = candidate.position
      const reached = await navigate(new goals.GoalNear(p.x, p.y, p.z, 2), 18000, signal)
      if (reached && reachableLog()) { searchMemory.reached(); return 'log_reached' }
      searchMemory.failed(candidate)
      return 'log_path_blocked'
    }
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    const index = explorationIndex++
    const [dx, dz] = directions[index % directions.length]
    const local = index % 2 === 0
    const radius = local ? 4 : 8 + 8 * (Math.floor(index / directions.length) % 4)
    const origin = local ? bot.entity.position : spawn
    const target = { x: origin.x + dx * radius, z: origin.z + dz * radius }
    if (!inBounds(target)) return 'exploration_boundary'
    const reached = await navigate(new goals.GoalNearXZ(target.x, target.z, local ? 1 : 2), 18000, signal)
    return reached ? 'explored' : 'exploration_blocked'
  }
  async function chop (signal) {
    let target = reachableLog()
    if (!target) return 'no_reachable_log'
    const p = target.position.clone()
    const before = inventoryCounts(inventory()).logs
    if (p.distanceTo(bot.entity.position) > 2) await navigate(new goals.GoalNear(p.x, p.y, p.z, 1.5), 10000, signal)
    target = bot.blockAt(p)
    if (!target?.name.endsWith('_log') || !bot.canSeeBlock(target) || !bot.canDigBlock(target)) return 'log_unreachable'
    await bot.dig(target)
    check(signal)
    // A high trunk block drops downward; the pickup waypoint stays at the
    // current standing level instead of trying to climb the removed block.
    await navigate(new goals.GoalNear(p.x, Math.floor(bot.entity.position.y), p.z, 0.7), 8000, signal)
    const deadline = Date.now() + 3000
    while (inventoryCounts(inventory()).logs <= before && Date.now() < deadline && bot.health > 0) {
      check(signal)
      await sleep(100)
    }
    return inventoryCounts(inventory()).logs > before ? 'log_collected' : 'log_broken_without_pickup'
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
    if (request.op === 'observe') return observe()
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

module.exports = { parseArgs, validateAction, inventoryCounts, validatePhysicalState, craftingRecipe, safeMovements, SearchMemory, replaceableForTable, ACTIONS, BACKEND }
if (require.main === module) main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exit(1) })
