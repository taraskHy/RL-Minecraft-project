'use strict'
// Runs a trained parkour policy in real Minecraft through Mineflayer.
//
//   node parkour_bot.js --policy P.json --courses C.json --out episodes.jsonl
//        [--port 25570] [--origin 0,80,0] [--pad -20,100,-20] [--viewer NAME] [--stall 200] [--max-ticks 3000]
//
// For each course: move to the spawn pad, clear the course area, build the
// platforms with /fill, /tp onto the start block, then drive the bot with the
// policy one game tick at a time. The policy sees what the game shows (the
// bot's position and velocity and the blocks around it) and acts through
// ordinary movement controls only. Every tick is logged so parkour.live can
// replay it in the simulator.

const fs = require('fs')
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const P = require('./policy')

function parseArgs (argv) {
  const a = { port: 25570, host: '127.0.0.1', origin: '0,80,0', pad: '-20,100,-20', stall: 200, maxTicks: 3000, username: 'ParkourBot', viewer: '' }
  for (let i = 2; i < argv.length; i += 2) {
    const k = argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    a[k] = argv[i + 1]
  }
  a.port = Number(a.port)
  a.stall = Number(a.stall)
  a.maxTicks = Number(a.maxTicks)
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

const bot = mineflayer.createBot({ host: args.host, port: args.port, username: args.username, version: '1.21.4', auth: 'offline' })

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
let episode = null
let finish = null
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

function applyAction (a) {
  const c = P.controls(a)
  bot.entity.yaw = P.wrapYaw(bot.entity.yaw + c.yawStep)
  for (const k of ['forward', 'back', 'left', 'right', 'jump', 'sprint']) bot.setControlState(k, c[k])
}

bot.on('physicsTick', () => {
  sanitizeAttributes()
  if (!episode) return
  const a = episode.step(state())
  if (a === null) {
    bot.clearControlStates()
    const done = finish
    finish = null
    if (done) done()
    return
  }
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
    if (Date.now() - t0 > ms) throw new Error('timed out waiting for ' + what)
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
  for (const [x0, x1, z0, z1, y] of c.platforms) {
    await chatCommand(`/fill ${x0 + OX} ${y + OY} ${z0 + OZ} ${x1 + OX} ${y + OY} ${z1 + OZ} minecraft:stone`)
  }
  const [sx, sy, sz] = c.spawn
  await chatCommand(`/tp ${args.username} ${sx + OX} ${sy + OY} ${sz + OZ} ${toNotchianYawDeg(c.spawn_yaw).toFixed(4)} 0`)
  await waitFor(() => {
    const p = bot.entity.position
    return Math.abs(p.x - (sx + OX)) < 1e-6 && Math.abs(p.z - (sz + OZ)) < 1e-6 && Math.abs(p.y - (sy + OY)) < 1e-6
  }, 10000, 'teleport')
  await waitFor(() => c.platforms.every(([x0, x1, z0, z1, y]) => isSolid(x0, y, z0) && isSolid(x1, y, z1)), 10000, 'course blocks')
  await sleep(500) // a few ticks to settle on the start block
  bot.entity.yaw = c.spawn_yaw
  corrections = 0
  const t0 = Date.now()
  const ep = new P.Episode(policy, c.platforms, isSolid, { stall: args.stall, maxTicks: args.maxTicks })
  await new Promise(resolve => { finish = resolve; episode = ep })
  episode = null
  const rec = ep.record({ index, level: c.level, seed: c.seed, wall_seconds: (Date.now() - t0) / 1000, server_corrections: corrections, mode: 'live' })
  out.write(JSON.stringify(rec) + '\n')
  log(`course ${index} (level ${c.level}): ${ep.result}, platform ${ep.cp}/${c.platforms.length - 1}, ${ep.ticks} ticks, corrections ${corrections}`)
  return rec
}

bot.once('spawn', async () => {
  try {
    log(`connected; ${courses.length} courses; origin ${args.origin}`)
    await sleep(1500)
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
