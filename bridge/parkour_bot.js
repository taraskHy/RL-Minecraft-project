'use strict'
// Runs a trained parkour policy in real Minecraft through Mineflayer.
//
//   node parkour_bot.js --policy P.json --courses C.json --out episodes.jsonl
//        [--port 25570] [--origin 0,80,0] [--pad -20,100,-20] [--viewer NAME] [--stall 200] [--max-ticks 3000]
//        [--assist CONFIG_JSON --assist-python PYTHON --assist-checkpoint CKPT --assist-root DIR --assist-device cpu|cuda]
//
// For each course: move to the spawn pad, clear the course area, build the
// platforms with /fill, /tp onto the start block, then drive the bot with the
// policy one game tick at a time. The policy sees what the game shows (the
// bot's position and velocity and the blocks around it) and acts through
// ordinary movement controls only. Every tick is logged so parkour.live can
// replay it in the simulator. With --assist (bridge/assist.js) the bot stops
// its physics on a planned takeoff while the Python planner searches a plan
// from its exact state (the server sees it stand still), checks the plan with
// prismarine-physics against the real blocks, and plays it.

const fs = require('fs')
require('./physics_patch') // vanilla collision tolerance, as in the simulator
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const P = require('./policy')
const A = require('./assist')
const { Physics } = require('prismarine-physics')
const path = require('path')
// Shaped platforms (Course.shapes): the block state of each shape name, as /setblock writes it.
const SHAPES = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'configs', 'block_shapes.json'), 'utf8'))
  .shapes.map(s => [s.name, s]))
function stateString (name) {
  const s = SHAPES[name]
  const props = Object.entries(s.properties).map(([k, v]) => `${k}=${v}`).join(',')
  return `minecraft:${s.block}` + (props ? `[${props}]` : '')
}
function sameBoxes (a, b) {
  return a.length === b.length && a.every((q, i) => q.length === b[i].length && q.every((v, j) => Math.abs(v - b[i][j]) < 1e-9))
}

function parseArgs (argv) {
  const a = { port: 25570, host: '127.0.0.1', origin: '0,80,0', pad: '-20,100,-20', stall: 200, maxTicks: 3000, username: 'ParkourBot', viewer: '', style: 'stone', webViewer: 0, hold: 0, planner: '', fallRule: 'local', assist: '' }
  for (let i = 2; i < argv.length; i += 2) {
    const k = argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    a[k] = argv[i + 1]
  }
  a.port = Number(a.port)
  a.stall = Number(a.stall)
  a.maxTicks = Number(a.maxTicks)
  a.webViewer = Number(a.webViewer)
  a.hold = Number(a.hold)
  a.origin = a.origin.split(',').map(Number)
  a.pad = a.pad.split(',').map(Number)
  return a
}

const args = parseArgs(process.argv)
const policy = P.Policy.load(args.policy)
const courses = JSON.parse(fs.readFileSync(args.courses, 'utf8')).courses
const out = fs.createWriteStream(args.out, { flags: 'a' })
const [OX, OY, OZ] = args.origin
const GRID = [64, 32, 64]

// Block looks. Every choice is a full cube with Minecraft's default friction (no ice,
// slime, soul sand or honey), so the bot moves exactly as it does on stone.
const RAINBOW = ['white_concrete', 'orange_concrete', 'magenta_concrete', 'light_blue_concrete', 'yellow_concrete',
  'lime_concrete', 'pink_concrete', 'cyan_concrete', 'purple_concrete', 'blue_concrete', 'red_concrete', 'green_concrete']
function blockFor (k, n, ice, slime) {
  if ((ice || []).includes(k)) return 'minecraft:ice' // the bot's physics reads the block type: ice friction
  if ((slime || []).includes(k)) return 'minecraft:slime_block' // and bounces off slime
  if (args.style !== 'rainbow') return 'minecraft:stone'
  if (k === 0) return 'minecraft:emerald_block'
  if (k === n - 1) return 'minecraft:gold_block'
  return 'minecraft:' + RAINBOW[(k - 1) % RAINBOW.length]
}
const OBSTACLE_BLOCK = () => args.style === 'rainbow' ? 'minecraft:quartz_block' : 'minecraft:stone'

const bot = mineflayer.createBot({ host: args.host, port: args.port, username: args.username, version: '1.21.4', auth: 'offline' })

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
let episode = null
let finish = null
let previous = [] // solid cells of the last course, to confirm they were cleared
let attributeNote = null
let corrections = 0

function log (msg) { process.stdout.write(`[bot] ${msg}\n`) }

// prismarine-physics adds its own sprint speed modifier but keeps the one a
// 1.21 server reports as well (the two have different ids), which would make
// the bot sprint at 1.3 x 1.3 speed. A vanilla client replaces the server's
// modifier with its own. Removing the server-sent movement speed attribute
// makes prismarine use its default (base 0.1 plus its own sprint modifier):
// the vanilla behaviour, and what the simulator models.
function sanitizeAttributes () {
  const attrs = bot.entity && bot.entity.attributes
  if (!attrs) return
  for (const key of Object.keys(attrs)) {
    if (key.includes('movement_speed') || key.includes('movementSpeed')) {
      if (!attributeNote) {
        attributeNote = { key, value: attrs[key].value, modifiers: attrs[key].modifiers }
        log('server movement attribute (removed each tick): ' + JSON.stringify(attributeNote))
      }
      delete attrs[key]
    }
  }
}

function state () {
  const e = bot.entity
  return {
    pos: [e.position.x - OX, e.position.y - OY, e.position.z - OZ],
    vel: [e.velocity.x, e.velocity.y, e.velocity.z],
    yaw: e.yaw,
    onGround: !!e.onGround,
    jumpTicks: bot.jumpTicks || 0,
    collidedH: !!e.isCollidedHorizontally
  }
}

function isSolid (x, y, z) {
  const b = bot.blockAt(new Vec3(x + OX, y + OY, z + OZ), false)
  return !!b && b.boundingBox === 'block'
}

// The top of a solid cell's collision shape as the bot sees it (1 for a full block).
function topOf (x, y, z) {
  const b = bot.blockAt(new Vec3(x + OX, y + OY, z + OZ), false)
  return b && b.shapes.length ? Math.max(...b.shapes.map(q => q[4])) : 1
}

function applyAction (a) {
  const c = P.controls(a)
  bot.entity.yaw = P.wrapYaw(bot.entity.yaw + c.yawStep)
  // When filming from the bot's eyes, look a little down so the next platforms are in view.
  // Pitch does not enter the movement physics (only elytra flight uses it).
  if (args.webViewer && args.firstPerson === 'true') bot.entity.pitch = -0.42
  for (const k of ['forward', 'back', 'left', 'right', 'jump', 'sprint']) bot.setControlState(k, c[k])
}

const assistConfig = args.assist ? JSON.parse(args.assist) : null
const service = assistConfig
  ? new A.PlannerService(args.assistPython, A.serviceArgs(assistConfig, args.assistCheckpoint, args.courses, args.assistDevice)
    .concat(['--realtime']), args.assistRoot) // real time: the GPU watchdog leaves this planner running
  : null
let planning = false
let courseIndex = -1

function endEpisode () {
  bot.clearControlStates()
  const done = finish
  finish = null
  if (done) done()
}

// The bot stands still (no physics ticks) while the planner searches; the plan is checked in world
// coordinates against the real blocks, as the bot's own physics will play it.
function planThenAct (ep) {
  planning = true
  bot.physicsEnabled = false
  const q = ep.pending
  const c = courses[courseIndex]
  service.plan({ index: courseIndex, cp: q.cp, target: q.target, kind: q.kind, attempt: q.attempt, state: q.state, lastAction: q.lastAction })
    .then(r => {
      const e = bot.entity
      const here = { pos: [e.position.x, e.position.y, e.position.z], vel: [e.velocity.x, e.velocity.y, e.velocity.z], yaw: e.yaw, onGround: !!e.onGround, jumpTicks: bot.jumpTicks || 0, collidedH: !!e.isCollidedHorizontally }
      const plats = c.platforms.map(([x0, x1, z0, z1, y]) => [x0 + OX, x1 + OX, z0 + OZ, z1 + OZ, y + OY])
      const extra = (c.extra || []).map(([x, y, z]) => [x + OX, y + OY, z + OZ])
      const geom = ep.geom ? ep.geom.map(([x0, x1, z0, z1, top]) => [x0 + OX, x1 + OX, z0 + OZ, z1 + OZ, top + OY]) : null
      const world = { getBlock: p => bot.blockAt(p, false) }
      const verified = !!r.plan && A.verifyPlan(Physics(bot.registry, world), world, here, r.plan, plats, q.cp, q.target, extra, geom)
      const brake = !r.plan && r.brake && A.verifyPlan(Physics(bot.registry, world), world, here, r.brake, plats, q.cp, q.cp, extra, geom, true) ? r.brake : null
      const a = ep.resume(verified ? r.plan : null, { found: !!r.plan, verified, rounds: r.rounds, seconds: r.seconds, error: r.error }, brake)
      if (a === P.PLAN) return planThenAct(ep) // a failed approach: the takeoff's own request, still frozen
      planning = false
      if (a === null) endEpisode()
      else applyAction(a)
      bot.physicsEnabled = true
    })
}

bot.on('physicsTick', () => {
  sanitizeAttributes()
  if (!episode || planning) return
  const a = episode.step(state())
  if (a === P.PLAN) return planThenAct(episode)
  if (a === null) return endEpisode()
  applyAction(a)
})

bot._client.on('position', () => {
  if (episode && episode.started && !episode.result) corrections++
})

async function chatCommand (cmd) {
  bot.chat(cmd)
  await sleep(30)
}

async function waitFor (cond, ms, what) {
  const t0 = Date.now()
  while (!cond()) {
    if (Date.now() - t0 > ms) {
      const e = bot.entity
      const below = bot.blockAt(e.position.offset(0, -0.5, 0), false)
      throw new Error(`timed out waiting for ${what}; bot at ${e.position} vel ${e.velocity} onGround ${e.onGround} ` +
        `block below ${below ? below.name : 'unloaded'}`)
    }
    await sleep(50)
  }
}

function toNotchianYawDeg (yaw) { return (Math.PI - yaw) * 180 / Math.PI }

async function runCourse (c, index) {
  bot.clearControlStates()
  episode = null
  await chatCommand(`/tp ${args.username} ${args.pad[0]} ${args.pad[1]} ${args.pad[2]}`)
  await sleep(200)
  for (let k = 0; k < GRID[1]; k += 8) {
    await chatCommand(`/fill ${OX} ${OY + k} ${OZ} ${OX + GRID[0] - 1} ${OY + k + 7} ${OZ + GRID[2] - 1} minecraft:air`)
  }
  if (args.style === 'rainbow') {
    // Remove the previous course's beacon (it stands below the course area).
    await chatCommand(`/fill ${OX - 4} ${OY - 7} ${OZ - 4} ${OX + GRID[0] + 3} ${OY - 6} ${OZ + GRID[2] + 3} minecraft:air`)
  }
  const shapes = c.shapes || {}
  for (const [k, [x0, x1, z0, z1, y]] of c.platforms.entries()) {
    if (shapes[k]) continue // shaped platforms last: fences and walls would connect to blocks placed after them
    await chatCommand(`/fill ${x0 + OX} ${y + OY} ${z0 + OZ} ${x1 + OX} ${y + OY} ${z1 + OZ} ${blockFor(k, c.platforms.length, c.ice, c.slime)}`)
  }
  const extra = c.extra || []
  for (const [x, y, z] of extra) await chatCommand(`/setblock ${x + OX} ${y + OY} ${z + OZ} ${OBSTACLE_BLOCK()}`)
  for (const [k, name] of Object.entries(shapes)) {
    const [x0, x1, z0, z1, y] = c.platforms[Number(k)]
    if (SHAPES[name].block.endsWith('_carpet')) {
      // Vanilla breaks a carpet with air below it at its next shape update, and a /fill of several carpets updates
      // each one. Structure void is not air but has no collision box (boundingBox 'empty'), so the cells under the
      // carpet stay empty for the physics and for isSolid, as in the simulator.
      await chatCommand(`/fill ${x0 + OX} ${y - 1 + OY} ${z0 + OZ} ${x1 + OX} ${y - 1 + OY} ${z1 + OZ} minecraft:structure_void replace minecraft:air`)
    }
    await chatCommand(`/fill ${x0 + OX} ${y + OY} ${z0 + OZ} ${x1 + OX} ${y + OY} ${z1 + OZ} ${stateString(name)}`)
  }
  const ladders = c.ladders || [] // after their walls (the obstacle cells): a ladder needs a block behind it
  for (const [x, y, z, f] of ladders) {
    await chatCommand(`/setblock ${x + OX} ${y + OY} ${z + OZ} ${stateString('ladder_' + ['north', 'east', 'south', 'west'][f])}`)
  }
  if (args.style === 'rainbow') {
    // A beacon beam rising beside the goal, from a beacon far below the course (beams have no collision).
    const used = new Set()
    for (const [x0, x1, z0, z1] of c.platforms) for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) used.add(`${x},${z}`)
    for (const [x, , z] of extra) used.add(`${x},${z}`)
    const g = c.platforms[c.platforms.length - 1]
    const cx = Math.floor((g[0] + g[1]) / 2); const cz = Math.floor((g[2] + g[3]) / 2)
    const spot = [[g[1] + 2, cz], [g[0] - 2, cz], [cx, g[3] + 2], [cx, g[2] - 2]].find(([x, z]) => !used.has(`${x},${z}`))
    if (spot) {
      const bx = spot[0] + OX; const bz = spot[1] + OZ; const by = OY - 6
      await chatCommand(`/fill ${bx - 1} ${by - 1} ${bz - 1} ${bx + 1} ${by - 1} ${bz + 1} minecraft:iron_block`)
      await chatCommand(`/setblock ${bx} ${by} ${bz} minecraft:beacon`)
    }
  }
  const [sx, sy, sz] = c.spawn
  // Always send decimals: Minecraft centres whole-number /tp coordinates on the block (12 -> 12.5).
  const coord = v => v.toFixed(6)
  await chatCommand(`/tp ${args.username} ${coord(sx + OX)} ${coord(sy + OY)} ${coord(sz + OZ)} ${toNotchianYawDeg(c.spawn_yaw).toFixed(4)} 0`)
  await waitFor(() => {
    const p = bot.entity.position
    return Math.abs(p.x - (sx + OX)) < 1e-6 && Math.abs(p.z - (sz + OZ)) < 1e-6 && Math.abs(p.y - (sy + OY)) < 1e-6
  }, 10000, 'teleport')
  await waitFor(() => c.platforms.every(([x0, x1, z0, z1, y]) => isSolid(x0, y, z0) && isSolid(x1, y, z1)), 10000, 'course blocks')
  await waitFor(() => extra.every(([x, y, z]) => isSolid(x, y, z)), 10000, 'obstacle blocks')
  // Every shaped platform must be exactly its block state (the physics collides with its shape), or the same block
  // with the same collision boxes: at noon a daylight detector's power follows the sky light within a second, which
  // changes its state but not its shape.
  await waitFor(() => Object.entries(shapes).every(([k, name]) => {
    const [x0, , z0, , y] = c.platforms[Number(k)]
    const b = bot.blockAt(new Vec3(x0 + OX, y + OY, z0 + OZ), false)
    return !!b && (b.stateId === SHAPES[name].state || (b.name === SHAPES[name].block && sameBoxes(b.shapes, SHAPES[name].boxes)))
  }), 10000, 'shaped blocks')
  await waitFor(() => ladders.every(([x, y, z, f]) => {
    const b = bot.blockAt(new Vec3(x + OX, y + OY, z + OZ), false)
    return !!b && b.stateId === SHAPES['ladder_' + ['north', 'east', 'south', 'west'][f]].state
  }), 10000, 'ladders')
  const cells = []
  for (const [x0, x1, z0, z1, y] of c.platforms) for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) cells.push([x, y, z])
  cells.push(...extra)
  cells.push(...ladders.map(([x, y, z]) => [x, y, z]))
  const wanted = new Set(cells.map(k => k.join(',')))
  const stale = previous.filter(k => !wanted.has(k.join(',')))
  await waitFor(() => stale.every(([x, y, z]) => !isSolid(x, y, z)), 10000, 'previous course cleared')
  previous = cells
  await sleep(500 + args.hold) // a few ticks to settle on the start block (longer when filming)
  bot.entity.yaw = c.spawn_yaw
  corrections = 0
  const t0 = Date.now()
  if (args.webViewer) log(`film: start ${index} ${t0}`) // wall-clock marks for cutting a video
  const ep = new P.Episode(policy, c.platforms, isSolid, {
    stall: args.stall,
    maxTicks: args.maxTicks,
    extra: c.extra || [],
    planner: args.planner ? JSON.parse(args.planner) : null,
    fallRule: args.fallRule,
    assist: assistConfig ? { config: assistConfig, takeoffs: c.assist_takeoffs || [], targets: c.assist_targets || null, retryHorizons: c.assist_retry_horizons || null, approach: c.assist_approach || null } : null,
    shapes: c.shapes || null,
    topOf
  })
  courseIndex = index
  await new Promise(resolve => { finish = resolve; episode = ep })
  episode = null
  if (args.webViewer) log(`film: end ${index} ${Date.now()}`)
  const rec = ep.record({ index, level: c.level, seed: c.seed, wall_seconds: (Date.now() - t0) / 1000, server_corrections: corrections, mode: 'live' })
  out.write(JSON.stringify(rec) + '\n')
  log(`course ${index} (level ${c.level}): ${ep.result}, platform ${ep.cp}/${c.platforms.length - 1}, ${ep.ticks} ticks, corrections ${corrections}`)
  if (args.webViewer) await sleep(3000) // stay on the last platform for the camera
  return rec
}

bot.once('spawn', async () => {
  try {
    log(`connected; ${courses.length} courses; origin ${args.origin}`)
    await sleep(1500)
    if (args.webViewer) {
      // Optional: prismarine-viewer renders the bot's real world with Minecraft's textures at
      // http://localhost:PORT (third-person, following the bot). Display only.
      const Module = require('module')
      const resolve = Module._resolveFilename
      Module._resolveFilename = function (request, ...rest) {
        return request === 'canvas' ? require.resolve('./canvas_stub') : resolve.call(this, request, ...rest)
      }
      require('prismarine-viewer/lib/mineflayer')(bot, { port: args.webViewer, firstPerson: args.firstPerson === 'true', viewDistance: 4 })
      log(`web viewer on http://localhost:${args.webViewer}`)
    }
    if (args.viewer) {
      log(`waiting for viewer ${args.viewer} to join 127.0.0.1:${args.port} ...`)
      await waitFor(() => bot.players[args.viewer] && bot.players[args.viewer].entity !== undefined, 15 * 60 * 1000, 'viewer')
      await chatCommand(`/gamemode spectator ${args.viewer}`)
    }
    let ok = 0
    for (let i = 0; i < courses.length; i++) {
      const rec = await runCourse(courses[i], i)
      if (args.viewer) await chatCommand(`/spectate ${args.username} ${args.viewer}`)
      if (rec.success) ok++
    }
    log(`done: ${ok}/${courses.length} courses completed`)
    if (service) service.close()
    out.end()
    await sleep(500)
    bot.quit()
    process.exit(0)
  } catch (err) {
    log('error: ' + (err && err.stack ? err.stack : err))
    out.end()
    process.exit(2)
  }
})

bot.on('kicked', reason => { log('kicked: ' + JSON.stringify(reason)); process.exit(3) })
bot.on('error', err => { log('connection error: ' + err.message) })
bot.on('death', () => log('bot died'))
