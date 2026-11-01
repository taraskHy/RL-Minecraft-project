'use strict'
// The course builder's bot (parkour/builder.py): build mode on the local server, in real Minecraft.
//
//   node builder_bot.js --policy P.json --skill S.json --planner ROUTER_JSON --python PY --root DIR --out DIR
//        [--converter-args JSON_LIST] [--port 25570] [--viewer NAME]
//
// A player chats: !build (creative mode and the kit of modelled blocks), !start / !via / !goal (the block
// they stand on), !route (what the scan makes of the build), !run [router|list] (the bot plays it while they
// spectate), !stop, !clear, !help; !demo K builds course K of --demo (a courses file) itself and marks its start
// and goal, for a check without a builder. On !route and !run the bot scans every block around the marks with
// bot.blockAt and the converter (python -m parkour.builder --serve) turns them into a course, or names the
// blocks it cannot model. The bot plays as the live bot does (bridge/parkour_bot.js, whose details for the
// game's exact movement are copied here: /tp with decimals, the server's movement-speed attribute removed every
// tick, the yaw set directly): the final policy, and the neo skill on each neo's block, with the learned router
// choosing the platforms (no course list; not for shaped blocks, which the route planner does not see) or with
// the course list. Every run is written to --out, and the converter replays it in the simulator.

const fs = require('fs')
const path = require('path')
const readline = require('readline')
const { spawn } = require('child_process')
require('./physics_patch') // vanilla collision tolerance, as in the simulator
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const P = require('./policy')

// parkour/builder.py KIT (tests/test_builder.py checks the two lists match)
const KIT = ['stone', 'ice', 'slime_block', 'ladder', 'stone_slab', 'stone_stairs', 'oak_trapdoor', 'white_carpet',
  'oak_fence', 'cobblestone_wall', 'iron_bars', 'end_rod']
const SCAN = [64, 32, 64] // the scanned box (the simulator's grid), centred on the marks
const PAD = [-20, 100, -20] // parkour/server.py SPAWN_PAD: its 5x5 stone floor is left out of every scan
const DEMO_ORIGIN = [100, 80, 100] // where !demo builds (away from the pad and from the live runs' course area at 0 80 0)
// Shaped platforms and ladders for !demo: the block state of each shape name, as /setblock writes it (as parkour_bot.js).
const SHAPES = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'configs', 'block_shapes.json'), 'utf8'))
  .shapes.map(s => [s.name, s]))
function stateString (name) {
  const s = SHAPES[name]
  const props = Object.entries(s.properties).map(([k, v]) => `${k}=${v}`).join(',')
  return `minecraft:${s.block}` + (props ? `[${props}]` : '')
}

function parseArgs (argv) {
  const a = { port: 25570, host: '127.0.0.1', username: 'ParkourBot', viewer: '', stall: 200, maxTicks: 4000, planner: '', skill: '', python: 'python', root: '..', out: '.', converterArgs: '[]', fallRule: 'local' }
  for (let i = 2; i < argv.length; i += 2) {
    const k = argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    a[k] = argv[i + 1]
  }
  a.port = Number(a.port)
  a.stall = Number(a.stall)
  a.maxTicks = Number(a.maxTicks)
  return a
}

const args = parseArgs(process.argv)
const policy = P.Policy.load(args.policy)
const skill = args.skill ? P.Policy.load(args.skill) : null
const routerPlanner = args.planner ? JSON.parse(args.planner) : null
const demos = args.demo ? JSON.parse(fs.readFileSync(args.demo, 'utf8')).courses : []
let expect = null // the demo course the marks belong to: the converter checks the scan gives it back
fs.mkdirSync(args.out, { recursive: true })
const runsLog = fs.createWriteStream(path.join(args.out, 'runs.jsonl'), { flags: 'a' })

// One JSON request per line to the converter, one answer per line back (parkour/builder.py serve).
class JsonService {
  constructor (python, argv, cwd) {
    this.proc = spawn(python, ['-m', 'parkour.builder', '--serve', ...argv], { cwd, stdio: ['pipe', 'pipe', 'inherit'] })
    this.waiting = []
    this.dead = null
    readline.createInterface({ input: this.proc.stdout }).on('line', line => {
      const w = this.waiting.shift()
      if (w) w(JSON.parse(line))
    })
    const die = why => {
      this.dead = this.dead || why
      for (const w of this.waiting.splice(0)) w({ error: this.dead })
    }
    this.proc.on('exit', code => die(`the converter exited (${code})`))
    this.proc.on('error', err => die(`the converter failed: ${err.message}`))
  }

  ask (req) {
    if (this.dead) return Promise.resolve({ error: this.dead })
    return new Promise(resolve => {
      this.waiting.push(resolve)
      this.proc.stdin.write(JSON.stringify(req) + '\n')
    })
  }

  close () { this.proc.stdin.end() }
}

const converter = new JsonService(args.python, JSON.parse(args.converterArgs), args.root)
const bot = mineflayer.createBot({ host: args.host, port: args.port, username: args.username, version: '1.21.4', auth: 'offline' })
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
let OX = 0; let OY = 0; let OZ = 0 // the world position of the course grid's cell (0, 0, 0), per run
let episode = null
let finish = null
let corrections = 0
let attributeNote = null
let busy = false // a scan or a run is under way
const marks = { start: null, via: [], goal: null }
let runs = 0

function log (msg) { process.stdout.write(`[builder] ${msg}\n`) }
function say (msg) { log(msg); bot.chat(msg.slice(0, 250)) }

// As bridge/parkour_bot.js: a vanilla client replaces the server's sprint modifier with its own, so the server's
// movement-speed attribute is removed every tick and prismarine-physics uses its default (what the simulator models).
function sanitizeAttributes () {
  const attrs = bot.entity && bot.entity.attributes
  if (!attrs) return
  for (const key of Object.keys(attrs)) {
    if (key.includes('movement_speed') || key.includes('movementSpeed')) {
      if (!attributeNote) {
        attributeNote = { key, value: attrs[key].value }
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

function topOf (x, y, z) {
  const b = bot.blockAt(new Vec3(x + OX, y + OY, z + OZ), false)
  return b && b.shapes.length ? Math.max(...b.shapes.map(q => q[4])) : 1
}

function applyAction (a) {
  const c = P.controls(a)
  bot.entity.yaw = P.wrapYaw(bot.entity.yaw + c.yawStep)
  for (const k of ['forward', 'back', 'left', 'right', 'jump', 'sprint']) bot.setControlState(k, c[k])
}

function endEpisode () {
  bot.clearControlStates()
  const done = finish
  finish = null
  if (done) done()
}

let lastCp = -1
let skillOn = false
bot.on('physicsTick', () => {
  sanitizeAttributes()
  if (!episode) return
  const a = episode.step(state())
  if (a === null) return endEpisode()
  applyAction(a)
  if (episode.cp !== lastCp) { // progress in chat
    lastCp = episode.cp
    if (episode.cp > 0 && episode.cp < episode.platforms.length - 1) bot.chat(`platform ${episode.cp}/${episode.platforms.length - 1}`)
  }
  const on = !!episode.skill && (episode.skill.cps ? episode.skill.cps.has(episode.cp) : episode.skill.left > 0)
  if (on && !skillOn) bot.chat('the neo skill takes over')
  skillOn = on
})

bot._client.on('position', () => {
  if (episode && episode.started && !episode.result) corrections++
})

async function command (cmd) {
  bot.chat(cmd)
  await sleep(40)
}

async function waitFor (cond, ms, what) {
  const t0 = Date.now()
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error(`timed out waiting for ${what}`)
    await sleep(50)
  }
}

function toNotchianYawDeg (yaw) { return (Math.PI - yaw) * 180 / Math.PI }

// The block a player stands on: below their feet, the cell whose collision top is level with them.
function blockUnder (name) {
  const p = bot.players[name] && bot.players[name].entity
  if (!p) return null
  for (const [dx, dz] of [[0, 0], [0.3, 0], [-0.3, 0], [0, 0.3], [0, -0.3], [0.3, 0.3], [-0.3, -0.3], [0.3, -0.3], [-0.3, 0.3]]) {
    const x = Math.floor(p.position.x + dx); const z = Math.floor(p.position.z + dz)
    for (let y = Math.floor(p.position.y); y >= Math.floor(p.position.y) - 2; y--) {
      const b = bot.blockAt(new Vec3(x, y, z), false)
      if (b && b.shapes.length && Math.abs(y + Math.max(...b.shapes.map(q => q[4])) - p.position.y) < 0.02) return [x, y, z]
    }
  }
  return null
}

// Every block in the box around the marks that is not air (the spawn pad left out), as the converter reads them.
async function scan () {
  const pts = [marks.start, marks.goal, ...marks.via]
  const lo = [0, 1, 2].map(a => Math.min(...pts.map(p => p[a])))
  const hi = [0, 1, 2].map(a => Math.max(...pts.map(p => p[a])))
  const box = [0, 1, 2].map(a => Math.floor((lo[a] + hi[a]) / 2 - SCAN[a] / 2))
  await command(`/tp ${args.username} ${marks.start[0] + 0.5} ${marks.start[1] + 3} ${marks.start[2] + 0.5}`)
  await waitFor(() => bot.entity.position.distanceTo(new Vec3(marks.start[0] + 0.5, marks.start[1] + 3, marks.start[2] + 0.5)) < 1, 10000, 'the teleport')
  const corners = [0, 1].flatMap(i => [0, 1].map(j => new Vec3(box[0] + i * (SCAN[0] - 1), marks.start[1], box[2] + j * (SCAN[2] - 1))))
  await waitFor(() => !!bot.blockAt(new Vec3(...marks.start), false) && corners.every(v => !!bot.blockAt(v, false)), 20000,
    'the chunks around the build') // (blockAt gives null in a chunk not loaded yet)
  await sleep(500)
  const blocks = []
  for (let x = box[0]; x < box[0] + SCAN[0]; x++) {
    for (let y = box[1]; y < box[1] + SCAN[1]; y++) {
      for (let z = box[2]; z < box[2] + SCAN[2]; z++) {
        if (y === PAD[1] - 1 && Math.abs(x - PAD[0]) <= 2 && Math.abs(z - PAD[2]) <= 2) continue
        const b = bot.blockAt(new Vec3(x, y, z), false)
        if (!b || b.name === 'air' || b.name === 'cave_air' || b.name === 'void_air') continue
        blocks.push([x, y, z, b.name, b.getProperties(), b.shapes])
      }
    }
  }
  return blocks
}

async function convert () {
  if (!marks.start || !marks.goal) {
    say('first stand on the first block and say !start, on the last and say !goal')
    return null
  }
  const blocks = await scan()
  const r = await converter.ask({ blocks, start: marks.start, goal: marks.goal, via: marks.via, name: `build ${runs + 1}`,
    ...(expect ? { expect } : {}) })
  if (r.error) {
    say(`cannot play this build: ${r.error}`)
    for (const [x, y, z, name, why] of (r.refused || []).slice(0, 5)) say(`  ${name} at ${x} ${y} ${z}: ${why}`)
    return null
  }
  for (const w of r.warnings) say(`note: ${w}`)
  if ('matches_expected' in r) say(r.matches_expected ? `the scan gives demo course ${expect.index} back exactly` : `the scan differs from demo course ${expect.index}`)
  return r
}

// !demo K: course K of --demo built at DEMO_ORIGIN with the live bot's commands (bridge/parkour_bot.js runCourse), its
// first and last platforms marked.
async function buildDemo (k) {
  const c = demos[k]
  if (!c) { say(`no demo course ${k} (${demos.length} in --demo)`); return }
  const [X, Y, Z] = DEMO_ORIGIN
  await command(`/tp ${args.username} ${X + 32} ${Y + 34} ${Z + 32}`)
  await sleep(1000)
  for (let y = 0; y < SCAN[1]; y += 8) await command(`/fill ${X} ${Y + y} ${Z} ${X + 63} ${Y + y + 7} ${Z + 63} minecraft:air`)
  const shapes = c.shapes || {}
  for (const [i, [x0, x1, z0, z1, y]] of c.platforms.entries()) {
    if (shapes[i]) continue
    const block = (c.ice || []).includes(i) ? 'minecraft:ice' : (c.slime || []).includes(i) ? 'minecraft:slime_block' : 'minecraft:stone'
    await command(`/fill ${x0 + X} ${y + Y} ${z0 + Z} ${x1 + X} ${y + Y} ${z1 + Z} ${block}`)
  }
  for (const [x, y, z] of c.extra || []) await command(`/setblock ${x + X} ${y + Y} ${z + Z} minecraft:stone`)
  for (const [i, name] of Object.entries(shapes)) {
    const [x0, x1, z0, z1, y] = c.platforms[Number(i)]
    if (SHAPES[name].block.endsWith('_carpet')) {
      await command(`/fill ${x0 + X} ${y - 1 + Y} ${z0 + Z} ${x1 + X} ${y - 1 + Y} ${z1 + Z} minecraft:structure_void replace minecraft:air`)
    }
    await command(`/fill ${x0 + X} ${y + Y} ${z0 + Z} ${x1 + X} ${y + Y} ${z1 + Z} ${stateString(name)}`)
  }
  for (const [x, y, z, f] of c.ladders || []) {
    await command(`/setblock ${x + X} ${y + Y} ${z + Z} ${stateString('ladder_' + ['north', 'east', 'south', 'west'][f])}`)
  }
  const first = c.platforms[0]; const last = c.platforms[c.platforms.length - 1]
  marks.start = [first[0] + X, first[4] + Y, first[2] + Z]
  marks.goal = [last[0] + X, last[4] + Y, last[2] + Z]
  marks.via = []
  expect = { index: k, course: c, origin: DEMO_ORIGIN }
  await waitFor(() => !!bot.blockAt(new Vec3(...marks.start), false) && bot.blockAt(new Vec3(...marks.start), false).name !== 'air', 10000, 'the demo blocks')
  say(`demo course ${k} built at ${X} ${Y} ${Z} (${c.platforms.length - 1} jumps); start and goal marked: say !run`)
}

function describe (r) {
  const a = r.about
  const bits = [`${a.jumps} jumps`, `${a.neos.length} neo${a.neos.length === 1 ? '' : 's'}`]
  for (const k of ['ice', 'slime', 'shapes', 'ladders']) if (a[k]) bits.push(`${a[k]} ${k === 'shapes' ? 'shaped' : k}`)
  return bits.join(', ')
}

async function playCourse (r, mode, player) {
  const c = r.course;
  [OX, OY, OZ] = r.origin
  const [sx, sy, sz] = c.spawn
  const coord = v => v.toFixed(6) // decimals: Minecraft centres whole-number /tp coordinates on the block
  const back = bot.players[player] && bot.players[player].entity ? bot.players[player].entity.position.clone() : null
  await command(`/gamemode spectator ${player}`)
  await command(`/tp ${args.username} ${coord(sx + OX)} ${coord(sy + OY)} ${coord(sz + OZ)} ${toNotchianYawDeg(c.spawn_yaw).toFixed(4)} 0`)
  await waitFor(() => {
    const p = bot.entity.position
    return Math.abs(p.x - (sx + OX)) < 1e-6 && Math.abs(p.z - (sz + OZ)) < 1e-6 && Math.abs(p.y - (sy + OY)) < 1e-6
  }, 10000, 'the teleport')
  await waitFor(() => c.platforms.every(([x0, x1, z0, z1, y]) => isSolid(x0, y, z0) && isSolid(x1, y, z1)), 10000, 'the course blocks')
  await command(`/spectate ${args.username} ${player}`)
  await sleep(500) // a few ticks to settle on the start block, as the live bot does
  bot.entity.yaw = c.spawn_yaw
  corrections = 0
  lastCp = -1
  skillOn = false
  const planner = mode === 'router' ? routerPlanner : null
  const ep = new P.Episode(policy, c.platforms, isSolid, {
    stall: args.stall,
    maxTicks: args.maxTicks,
    extra: c.extra || [],
    planner,
    fallRule: args.fallRule,
    shapes: c.shapes || null,
    topOf,
    skill: skill ? { policy: skill, cps: c.skill_cps } : null
  })
  const t0 = Date.now()
  await new Promise(resolve => { finish = resolve; episode = ep })
  episode = null
  const rec = ep.record({ index: 0, mode, wall_seconds: (Date.now() - t0) / 1000, server_corrections: corrections })
  runsLog.write(JSON.stringify({ run: runs, course: c, origin: r.origin, mode, episode: rec }) + '\n')
  const n = c.platforms.length - 1
  if (ep.result === 'success') say(`made it: ${n} jumps in ${ep.ticks} ticks (${(ep.ticks / 20).toFixed(1)} s), ${corrections} server corrections`)
  else if (ep.result === 'fell') say(`fell after platform ${ep.cp} of ${n}${r.about.neos.includes(ep.cp) ? ' (at a neo)' : ''}`)
  else say(`stopped at platform ${ep.cp} of ${n}: no progress for ${args.stall} ticks`)
  await command(`/spectate ${player}`) // (no target: stop spectating)
  await command(`/gamemode creative ${player}`)
  if (back) await command(`/tp ${player} ${coord(back.x)} ${coord(back.y)} ${coord(back.z)}`)
  const check = await converter.ask({ check: { course: c, episode: rec, mode } })
  if (check.error) say(`simulator check failed: ${check.error}`)
  else {
    say(`simulator replay of the game's ticks: largest difference ${check.max_pos_err.toExponential(1)} blocks; ` +
      `the simulator's own run: ${check.sim_success ? 'made it' : 'did not make it'} in ${check.sim_ticks} ticks` +
      (check.sim_success === rec.success && check.sim_ticks === rec.ticks ? ' (the same)' : ' (different)'))
  }
  runs++
}

const HELP = ['!build: creative mode and the kit', '!start / !goal: stand on the first / last block', '!via: an optional platform the route must pass, in order',
  '!route: what the bot makes of the build', '!run [router|list]: the bot plays it, you watch', '!stop, !clear, !help']

async function onCommand (player, line) {
  const [cmd, opt] = line.trim().split(/\s+/)
  if (cmd === '!help') { for (const h of HELP) say(h); return }
  if (cmd === '!stop') { if (episode) { episode.result = 'truncated'; endEpisode(); say('stopped') } return }
  if (busy) { say('busy; !stop ends a run'); return }
  if (cmd === '!build') {
    await command(`/gamemode creative ${player}`)
    await command(`/clear ${player}`)
    for (const k of KIT) await command(`/give ${player} minecraft:${k} 64`)
    say('the kit: only blocks the simulator models. Platforms one block thick, a block of air between them, ' +
      'four blocks of air above. Walls of stone. Then !start, !goal, !run')
    return
  }
  if (cmd === '!start' || cmd === '!goal' || cmd === '!via') {
    const b = blockUnder(player)
    if (!b) { say('stand on a block first (near its middle)'); return }
    if (cmd === '!via') marks.via.push(b); else marks[cmd.slice(1)] = b
    expect = null
    say(`${cmd.slice(1)} at ${b.join(' ')}${cmd === '!via' ? ` (via ${marks.via.length})` : ''}`)
    return
  }
  if (cmd === '!clear') { marks.start = null; marks.goal = null; marks.via = []; expect = null; say('marks cleared'); return }
  if (cmd === '!demo') {
    busy = true
    try { await buildDemo(Number(opt || 0)) } catch (err) { say(`error: ${err.message}`) } finally { busy = false }
    return
  }
  if (cmd === '!route' || cmd === '!run') {
    busy = true
    try {
      const r = await convert()
      if (!r) return
      say(`the course: ${describe(r)}`)
      if (cmd === '!route') {
        r.course.platforms.forEach(([x0, x1, z0, z1, y], k) => {
          if (k < 16) setTimeout(() => command(`/particle minecraft:happy_villager ${(x0 + x1 + 1) / 2 + r.origin[0]} ${y + 1.5 + r.origin[1]} ${(z0 + z1 + 1) / 2 + r.origin[2]} 0.3 0.3 0.3 0 15`), 400 * k)
        })
        return
      }
      const mode = opt === 'list' ? 'list' : opt === 'router' ? 'router' : (r.about.router_ok && routerPlanner ? 'router' : 'list')
      if (mode === 'router' && (!r.about.router_ok || !routerPlanner)) { say('the learned router does not see shaped blocks: use !run list'); return }
      say(mode === 'router' ? 'playing with the learned router (no course list), the final policy and the neo skill'
        : 'playing with the course list, the final policy and the neo skill')
      await playCourse(r, mode, player)
    } catch (err) {
      say(`error: ${err.message}`)
    } finally {
      busy = false
    }
  }
}

bot.on('chat', (username, message) => {
  if (username === args.username || !message.startsWith('!')) return
  onCommand(username, message).catch(err => log('error: ' + (err && err.stack ? err.stack : err)))
})

bot.once('spawn', async () => {
  log(`connected on port ${args.port}; waiting for !build, !start, !goal, !run`)
  await sleep(1500)
  bot.chat('builder ready: say !help')
})

bot.on('kicked', reason => { log('kicked: ' + JSON.stringify(reason)); converter.close(); process.exit(3) })
bot.on('error', err => { log('connection error: ' + err.message) })
bot.on('end', () => { converter.close(); process.exit(0) })
