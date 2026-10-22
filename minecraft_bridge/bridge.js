'use strict'

// One JSON request per line, one JSON response per line. stdout is protocol only.
// This adapter deliberately implements ONLY the first development-arena task.
const readline = require('node:readline')

const SIZE = 5
const LOGS = [[1, 1], [3, 2]]
const LAVA = [2, 3]
const STATION = [4, 4]
const ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'CHOP', 'CRAFT_PLANKS', 'CRAFT_STICKS', 'CRAFT_PICKAXE']

function parseArgs (args) {
  const config = { host: '127.0.0.1', port: 25565, username: 'RLCourseBot', version: '1.21.4', devArena: false, origin: [0, 64, 0] }
  const fields = { '--host': 'host', '--port': 'port', '--username': 'username', '--version': 'version' }
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === '--dev-arena') config.devArena = true
    else if (arg === '--origin') {
      const values = args.slice(i + 1, i + 4).map(Number)
      if (values.length !== 3 || !values.every(Number.isInteger)) throw new Error('--origin requires three integers')
      config.origin = values
      i += 3
    } else if (fields[arg] && args[i + 1] !== undefined) config[fields[arg]] = args[++i]
    else throw new Error(`Unknown or incomplete option: ${arg}`)
  }
  config.port = Number(config.port)
  if (!Number.isInteger(config.port) || config.port < 1 || config.port > 65535) throw new Error('Invalid port')
  if (!['127.0.0.1', 'localhost', '::1'].includes(config.host)) throw new Error('This development bridge only connects to localhost')
  if (!/^[A-Za-z0-9_]{1,16}$/.test(config.username)) throw new Error('Invalid bot username')
  if (config.version !== '1.21.4') throw new Error('This development arena is pinned to Minecraft Java 1.21.4')
  const [x, y, z] = config.origin
  if (Math.abs(x) > 29999980 || Math.abs(z) > 29999980 || y < -60 || y > 310) throw new Error('Arena origin is outside supported world bounds')
  return config
}

function validateStart (start) {
  if (!Array.isArray(start) || start.length !== 2 || !start.every(Number.isInteger)) throw new Error('start must be [x,z] integers')
  const [x, z] = start
  if (x < 0 || x >= SIZE || z < 0 || z >= SIZE || (x === LAVA[0] && z === LAVA[1]) || LOGS.some(([lx, lz]) => lx === x && lz === z)) {
    throw new Error('Start must be a safe, empty arena cell')
  }
  return start
}

function gridPosition (position, origin, { allowDead = false } = {}) {
  if (![position.x, position.y, position.z].every(Number.isFinite)) throw new Error('Bot position contains a non-finite coordinate')
  const x = Math.floor(position.x - origin[0])
  const z = Math.floor(position.z - origin[2])
  if (x < 0 || x >= SIZE || z < 0 || z >= SIZE) throw new Error('Bot left arena bounds; reset is required')
  if (!allowDead) {
    if (Math.abs(position.y - origin[1]) > 0.35 || Math.abs(position.x - (origin[0] + x + 0.5)) > 0.35 || Math.abs(position.z - (origin[2] + z + 0.5)) > 0.35) {
      throw new Error('Bot pose does not match a cell center; reset is required')
    }
  }
  return [x, z]
}

function expectedArenaBlock (x, y, z) {
  if (x < -1 || x > 5 || z < -1 || z > 5 || y < -2 || y > 4) throw new Error('Outside development arena block bounds')
  if (y === -2) return 'stone'
  if (y === -1) return x === LAVA[0] && z === LAVA[1] ? 'lava' : 'stone'
  if (y === 4) return 'barrier'
  if (x === 5 && z === 4 && y === 0) return 'crafting_table'
  if (x === -1 || x === 5 || z === -1 || z === 5) return 'barrier'
  if (y === 0 && LOGS.some(([lx, lz]) => lx === x && lz === z)) return 'oak_log'
  return 'air'
}

function verifyArena (blockAt) {
  for (let x = -1; x <= 5; x++) {
    for (let z = -1; z <= 5; z++) {
      for (let y = -2; y <= 4; y++) {
        const expected = expectedArenaBlock(x, y, z)
        if (blockAt(x, y, z)?.name !== expected) throw new Error(`Arena reset mismatch at relative (${x},${y},${z}); expected ${expected}`)
      }
    }
  }
}

async function main () {
  const config = parseArgs(process.argv.slice(2))
  const mineflayer = require('mineflayer')
  const { Vec3 } = require('vec3')
  const bot = mineflayer.createBot({ host: config.host, port: config.port, username: config.username, version: config.version, auth: 'offline', hideErrors: true, respawn: false })
  let connected = false
  let initialized = false
  let ended = false
  let fatal = null
  const log = (message) => process.stderr.write(`[minecraft bridge] ${message}\n`)
  bot.on('spawn', () => { connected = true })
  bot.on('error', (err) => { fatal = err; log(err.message) })
  bot.on('kicked', (reason) => { fatal = new Error(`Kicked: ${JSON.stringify(reason)}`); log(fatal.message) })
  bot.on('end', (reason) => { connected = false; ended = true; log(`Connection ended: ${reason}`) })

  function check (signal) {
    if (signal.aborted) throw new Error('Operation timed out; bridge must be restarted')
    if (fatal) throw fatal
    if (ended) throw new Error('Minecraft connection closed')
  }
  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))
  async function until (predicate, ms, signal, message) {
    const deadline = Date.now() + ms
    while (!predicate()) {
      check(signal)
      if (Date.now() > deadline) throw new Error(message)
      await sleep(50)
    }
    check(signal)
  }
  async function ticks (count, signal) {
    check(signal)
    // waitForTicks can hang after a disconnect; the request deadline also bounds it.
    await bot.waitForTicks(count)
    check(signal)
  }
  function world (x, y, z) { return new Vec3(config.origin[0] + x, config.origin[1] + y, config.origin[2] + z) }
  function block (x, y, z) {
    const result = bot.blockAt(world(x, y, z))
    if (!result) throw new Error('Arena chunk is not loaded')
    return result
  }
  function inventory () {
    const counts = {}
    for (const item of bot.inventory.items()) counts[item.name] = (counts[item.name] || 0) + item.count
    return counts
  }
  function observe () {
    const dead = bot.health <= 0
    const [x, z] = gridPosition(bot.entity.position, config.origin, { allowDead: dead })
    let mask = 0
    for (let i = 0; i < LOGS.length; i++) {
      const current = block(LOGS[i][0], 0, LOGS[i][1])
      if (current.name === 'oak_log') mask |= (1 << i)
      else if (current.name !== 'air') throw new Error('Unexpected block at a log cell; reset is required')
    }
    const inv = inventory()
    return {
      state: [x, z, mask, inv.oak_log || 0, inv.oak_planks || 0, inv.stick || 0],
      health: bot.health,
      food: bot.food,
      position: { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z },
      inventory: inv,
      success: (inv.wooden_pickaxe || 0) > 0,
      death: dead,
      backend: 'minecraft-java-development-arena',
      game_version: bot.version
    }
  }

  async function moveToCell (x, z, signal, hazard = false) {
    const target = world(x + 0.5, 0, z + 0.5)
    try {
      for (let attempt = 0; attempt < 3; attempt++) {
        for (let t = 0; t < 60; t++) {
          check(signal)
          if (bot.health <= 0) return
          const pos = bot.entity.position
          const distance = Math.hypot(target.x - pos.x, target.z - pos.z)
          if (distance < 0.18) break
          await bot.lookAt(new Vec3(target.x, pos.y + (bot.entity.eyeHeight || 1.62), target.z), true)
          bot.setControlState('forward', true)
          await ticks(1, signal)
        }
        bot.clearControlStates()
        await ticks(6, signal)
        if (hazard) {
          const pos = bot.entity.position
          if (Math.floor(pos.x - config.origin[0]) === x && Math.floor(pos.z - config.origin[2]) === z) return
        } else {
          const pos = bot.entity.position
          // Bounds/height violations are not repairable by a horizontal move.
          // Ordinary lateral drift is retried, then checked against observed
          // coordinates; do not claim the requested cell merely from a command.
          const pose = gridPosition(pos, config.origin, { allowDead: true })
          if (Math.abs(pos.y - config.origin[1]) > 0.35) throw new Error('Movement left the arena floor; reset is required')
          if (pose[0] === x && pose[1] === z && Math.abs(pos.x - target.x) <= 0.35 && Math.abs(pos.z - target.z) <= 0.35) return
        }
      }
      throw new Error('Movement did not reach the requested cell')
    } finally { bot.clearControlStates() }
  }

  async function command (value, signal) {
    check(signal)
    bot.chat('/' + value)
    await ticks(3, signal)
  }
  async function reset (request, signal) {
    if (!config.devArena) throw new Error('reset modifies a dedicated world; start with --dev-arena to acknowledge this')
    if (bot.game.dimension !== 'overworld') throw new Error('Development arena reset requires the Overworld')
    const [sx, sz] = validateStart(request.start || [0, 0])
    initialized = false
    bot.clearControlStates()
    if (bot.health <= 0) {
      connected = false
      bot.respawn()
      await until(() => connected && bot.health > 0, 10000, signal, 'Could not respawn bot')
    }
    const [ox, oy, oz] = config.origin
    const box = (x1, y1, z1, x2, y2, z2) => `${ox + x1} ${oy + y1} ${oz + z1} ${ox + x2} ${oy + y2} ${oz + z2}`
    // All privileged changes are restricted to this explicit development reset.
    // The block region x/z [-1,5], y [-2,4] is replaced, including existing builds.
    const commands = [
      'gamerule doFireTick false',
      'gamerule doMobSpawning false',
      `gamemode creative ${config.username}`,
      `tp ${config.username} ${ox + sx + 0.5} ${oy + 2} ${oz + sz + 0.5}`,
      // Two solid layers contain the lava even above caves or an empty world.
      `fill ${box(-1, -2, -1, 5, -1, 5)} stone`,
      `fill ${box(-1, 0, -1, 5, 4, 5)} air`,
      `fill ${box(-1, 0, -1, -1, 3, 5)} barrier`,
      `fill ${box(5, 0, -1, 5, 3, 5)} barrier`,
      `fill ${box(0, 0, -1, 4, 3, -1)} barrier`,
      `fill ${box(0, 0, 5, 4, 3, 5)} barrier`,
      `fill ${box(-1, 4, -1, 5, 4, 5)} barrier`,
      `setblock ${ox + LAVA[0]} ${oy - 1} ${oz + LAVA[1]} lava`,
      ...LOGS.map(([x, z]) => `setblock ${ox + x} ${oy} ${oz + z} oak_log`),
      `setblock ${ox + 5} ${oy} ${oz + 4} crafting_table`,
      `kill @e[type=minecraft:item,x=${ox - 1},y=${oy - 1},z=${oz - 1},dx=6,dy=6,dz=6]`,
      `clear ${config.username}`,
      `effect clear ${config.username}`,
      `effect give ${config.username} minecraft:instant_health 1 10 true`,
      `effect give ${config.username} minecraft:saturation 1 10 true`,
      `gamemode survival ${config.username}`,
      `tp ${config.username} ${ox + sx + 0.5} ${oy} ${oz + sz + 0.5}`
    ]
    for (const value of commands) await command(value, signal)
    await ticks(20, signal)
    verifyArena(block)
    const result = observe()
    if (result.state[0] !== sx || result.state[1] !== sz || result.state[2] !== 3 || Object.keys(result.inventory).length || result.health < 20 || result.food < 20 || bot.game.gameMode !== 'survival') {
      throw new Error('Arena reset did not verify; check bot operator permission and use a dedicated Overworld server')
    }
    initialized = true
    return { ...result, event: 'reset' }
  }

  async function step (request, signal) {
    if (!initialized) throw new Error('A successful explicit development-arena reset is required before step')
    const action = request.action
    if (!Number.isInteger(action) || action < 0 || action >= ACTIONS.length) throw new Error('action must be an integer from 0 through 7')
    const before = observe()
    if (before.death || before.success) throw new Error('Episode has ended; reset before stepping')
    const [x, z, mask] = before.state
    let event = ACTIONS[action].toLowerCase()
    if (action < 4) {
      const [dx, dz] = [[0, -1], [0, 1], [1, 0], [-1, 0]][action]
      const tx = x + dx; const tz = z + dz
      const occupied = LOGS.some(([lx, lz], i) => lx === tx && lz === tz && (mask & (1 << i)))
      if (tx < 0 || tx >= SIZE || tz < 0 || tz >= SIZE || occupied) event = 'blocked'
      else {
        const hazard = tx === LAVA[0] && tz === LAVA[1]
        await moveToCell(tx, tz, signal, hazard)
        if (hazard) {
          await until(() => bot.health <= 0, 20000, signal, 'Lava did not kill the bot within the task timeout; episode invalid')
          event = 'death'
        }
      }
    } else if (action === 4) {
      const target = LOGS.find(([lx, lz], i) => (mask & (1 << i)) && Math.abs(lx - x) + Math.abs(lz - z) === 1)
      if (!target) event = 'no_adjacent_log'
      else {
        const targetBlock = block(target[0], 0, target[1])
        if (!bot.canDigBlock(targetBlock)) throw new Error('Adjacent log is not reachable for digging')
        await bot.dig(targetBlock)
        check(signal)
        // Collect the real item drop and return; never fabricate inventory updates.
        await moveToCell(target[0], target[1], signal)
        await until(() => (inventory().oak_log || 0) > before.state[3], 5000, signal, 'Log was broken but its item was not collected')
        await moveToCell(x, z, signal)
      }
    } else {
      const itemName = { 5: 'oak_planks', 6: 'stick', 7: 'wooden_pickaxe' }[action]
      const atStation = x === STATION[0] && z === STATION[1]
      if (action === 7 && !atStation) event = 'not_at_workbench'
      else {
        const item = bot.registry.itemsByName[itemName]
        const table = action === 7 ? block(5, 0, 4) : null
        const recipe = bot.recipesFor(item.id, null, 1, table)[0]
        if (!recipe) event = 'missing_materials'
        else {
          await bot.craft(recipe, 1, table)
          // 4.39.0 can optimistically change inventory for 2x2 crafting. This
          // pinned private method requests an authoritative server window sync.
          // It remains subject to the surrounding request's 40-second deadline.
          if (typeof bot._syncWindow !== 'function') throw new Error('Pinned Mineflayer inventory sync API is missing')
          await bot._syncWindow(bot.inventory)
          check(signal)
          if ((inventory()[itemName] || 0) <= (before.inventory[itemName] || 0)) throw new Error('Crafting returned without the expected inventory change')
        }
      }
    }
    return { ...observe(), event }
  }

  async function dispatch (request, signal) {
    if (!request || typeof request !== 'object' || Array.isArray(request)) throw new Error('Request must be a JSON object')
    if (request.op === 'close') return { closed: true }
    await until(() => connected && bot.entity && bot.inventory, 20000, signal, 'Minecraft connection did not spawn; check local server and version')
    if (request.op === 'status') return { connected, initialized, dev_arena: config.devArena, game_version: bot.version, backend: 'minecraft-java-development-arena' }
    if (request.op === 'reset') return reset(request, signal)
    if (request.op === 'step') return step(request, signal)
    if (request.op === 'observe') return observe()
    throw new Error(`Unknown operation: ${request.op}`)
  }
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity })
  let queue = Promise.resolve()
  let closing = false
  input.on('line', line => {
    queue = queue.then(async () => {
      if (closing) return
      let request
      const controller = new AbortController()
      let timeout
      try {
        request = JSON.parse(line)
        const deadline = new Promise((resolve, reject) => {
          timeout = setTimeout(() => {
            controller.abort()
            bot.clearControlStates()
            try { bot.stopDigging() } catch (_) {}
            reject(new Error('Minecraft operation exceeded 40 seconds; reconnect before retrying'))
          }, 40000)
        })
        const result = await Promise.race([dispatch(request, controller.signal), deadline])
        process.stdout.write(JSON.stringify({ id: request.id ?? null, ok: true, result }) + '\n')
        if (request.op === 'close') closing = true
      } catch (err) {
        initialized = false
        process.stdout.write(JSON.stringify({ id: request?.id ?? null, ok: false, error: err.message }) + '\n')
        if (controller.signal.aborted) closing = true
      } finally {
        clearTimeout(timeout)
        bot.clearControlStates()
        if (closing) { input.close(); bot.quit(); setTimeout(() => process.exit(0), 100) }
      }
    }).catch(err => { log(err.stack); process.exitCode = 1 })
  })
  input.on('close', () => {
    // EOF must not discard requests that were already read (for example when
    // a JSON-lines file is piped into the bridge). Finish the bounded queue.
    queue.finally(() => {
      closing = true
      bot.clearControlStates()
      bot.quit()
      setTimeout(() => process.exit(0), 150)
    })
  })
  process.on('SIGINT', () => { bot.clearControlStates(); bot.quit(); process.exit(0) })
}

module.exports = { parseArgs, validateStart, gridPosition, expectedArenaBlock, verifyArena, ACTIONS }
if (require.main === module) main().catch(err => { process.stderr.write(`${err.stack}\n`); process.exit(1) })
