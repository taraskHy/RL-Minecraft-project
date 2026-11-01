'use strict'
// The course builder's bot (parkour/builder.py): build mode on the local server, in real Minecraft.
//
//   node builder_bot.js --policy P.json --skill S.json --planner ROUTER_JSON --python PY --root DIR --out DIR
//        [--converter-args JSON_LIST] [--port 25570] [--viewer NAME]
//
// The same commands come three ways: the buttons of the panel beside the spawn pad (bridge/builder_panel.js:
// command blocks, for a player whose chat is off), a player's chat (!build, !route, ...) and a line typed in this
// program's console (build, route, ...; for --viewer, or the one player online). !build: creative mode and the kit
// of modelled blocks, the marker blocks with it; !start / !via / !goal: the block the player stands on; !route:
// what the scan makes of the build; !run [router|list]: the bot plays it while they spectate; !stop, !clear,
// !help; !demo [K] builds course K of --demo (a courses file; the next one without K) itself and marks its start
// and goal, for a check without a builder. Without !start and !goal (or after !clear) the marks are the marker
// blocks within MARKER_RANGE of the spawn pad: a gold block the start, a diamond block the goal, emerald blocks
// platforms the route must pass (nearest the start first); to the simulator they are stone. On !route and !run
// the bot scans every block in a box around the start wide enough for any course (BOX) and the converter (python
// -m parkour.builder --serve) keeps the blocks linked to the start's block, turns them into a course, or names the
// blocks it cannot model (the spawn pad and the panel are walls of a build near them). The bot plays as the live bot does
// (bridge/parkour_bot.js, whose details for the game's exact movement are copied here: /tp with decimals, the
// server's movement-speed attribute removed every tick, the yaw set directly): the final policy, and the neo
// skill on each neo's block, with the learned router choosing the platforms (no course list; not for shaped
// blocks, which the route planner does not see) or with the course list. Every run is written to --out, and the
// converter replays it in the simulator. The bot answers in system messages and on the action bar, which a player
// sees even with chat turned off, and in this console.

const fs = require('fs')
const path = require('path')
const readline = require('readline')
const { spawn } = require('child_process')
require('./physics_patch') // vanilla collision tolerance, as in the simulator
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const P = require('./policy')
const Panel = require('./builder_panel')

// parkour/builder.py KIT (tests/test_builder.py checks the two lists match)
const KIT = ['stone', 'gold_block', 'diamond_block', 'emerald_block', 'slime_block', 'ice', 'ladder', 'stone_slab', 'stone_stairs',
  'oak_trapdoor', 'white_carpet', 'oak_fence', 'cobblestone_wall', 'iron_bars', 'end_rod']
const MARKERS = { gold_block: 'start', diamond_block: 'goal', emerald_block: 'via' } // parkour/builder.py MARKERS
const MARKER_RANGE = 96 // marker blocks count within this many blocks of the spawn pad
const SCAN = [64, 32, 64] // the simulator's grid (what !demo clears)
const BOX = [62, 32, 62] // the scanned box's half-size around the start: a course is at most 60 blocks across, 32 rows
const PAD = Panel.PAD // parkour/server.py SPAWN_PAD
const HOME = new Vec3(PAD[0] + 0.5, PAD[1], PAD[2] + 0.5) // where the bot waits: it sees the panel and every marker in range
const PANEL = new Map(Panel.cells().map(([x, y, z, what]) => [`${x},${y},${z}`, what]))
const PANEL_BLOCK = { command_block: 'command_block', button: 'stone_button', sign: 'oak_sign' }
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
const marks = { start: null, via: [], goal: null, source: null } // source: 'chat' (!start, !goal, !via), 'demo', or null (the marker blocks)
let runs = 0
let nextDemo = 0 // the course the Demo button (!demo without K) builds next
let watcher = null // the player watching the current run

function log (msg) { process.stdout.write(`[builder] ${msg}\n`) }

// msg in pieces of at most n characters, cut at spaces where possible.
function pieces (msg, n) {
  const out = []
  let rest = String(msg)
  while (rest.length > n) {
    let cut = rest.lastIndexOf(' ', n)
    if (cut < n / 2) cut = n
    out.push(rest.slice(0, cut))
    rest = rest.slice(cut).replace(/^ /, '')
  }
  if (rest) out.push(rest)
  return out
}

const others = () => `@a[name=!${args.username}]` // every player but the bot
// System messages reach a player whose chat is turned off for players (Chat: Commands Only) and the action bar one
// whose chat is Hidden (a player's chat reaches neither), so what matters goes to both.
function tell (msg, target = null) {
  for (const part of pieces(msg, 150)) command(`/tellraw ${target || others()} ${JSON.stringify([{ text: '[builder] ', color: 'gold' }, { text: part }])}`)
}
function bar (msg, target = null) { command(`/title ${target || others()} actionbar ${JSON.stringify({ text: String(msg).slice(0, 120) })}`) }
const barQueue = [] // say()'s lines for the action bar: a piece every 2.5 s, so that each can be read
setInterval(() => { const m = barQueue.shift(); if (m) bar(m) }, 2500)
function say (msg, onBar = true) {
  log(msg)
  tell(msg)
  if (onBar) {
    barQueue.push(...pieces(msg, 100))
    barQueue.splice(0, Math.max(0, barQueue.length - 8)) // (the newest eight)
  }
}

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
  if (episode.cp !== lastCp) { // progress on the watcher's action bar
    lastCp = episode.cp
    if (episode.cp > 0 && episode.cp < episode.platforms.length - 1) bar(`platform ${episode.cp}/${episode.platforms.length - 1}`, watcher)
  }
  const on = !!episode.skill && (episode.skill.cps ? episode.skill.cps.has(episode.cp) : episode.skill.left > 0)
  if (on && !skillOn) bar('the neo skill takes over', watcher)
  skillOn = on
})

bot._client.on('position', () => {
  if (episode && episode.started && !episode.result) corrections++
})

async function command (cmd) {
  if (!bot.entity) { log(`not connected yet, not sent: ${cmd.slice(0, 80)}`); return }
  if (cmd.length > 256) { log(`over the 256 characters of a chat command, not sent: ${cmd.slice(0, 80)}...`); return }
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

// Every block that is not air in the box of half-size BOX around the start, as the converter reads them, and the
// box [x0, y0, z0, x1, y1, z1]: the converter keeps the blocks linked to the start's block, and refuses a build
// that reaches the box's faces (it would be larger than a course).
async function scan () {
  const s = marks.start
  const box = [s[0] - BOX[0], s[1] - BOX[1], s[2] - BOX[2], s[0] + BOX[0], s[1] + BOX[1], s[2] + BOX[2]]
  await command(`/tp ${args.username} ${s[0] + 0.5} ${s[1] + 3} ${s[2] + 0.5}`)
  await waitFor(() => bot.entity.position.distanceTo(new Vec3(s[0] + 0.5, s[1] + 3, s[2] + 0.5)) < 1, 10000, 'the teleport')
  const corners = [0, 3].flatMap(i => [2, 5].map(j => new Vec3(box[i], s[1], box[j])))
  await waitFor(() => !!bot.blockAt(new Vec3(...s), false) && corners.every(v => !!bot.blockAt(v, false)), 20000,
    'the chunks around the build') // (blockAt gives null in a chunk not loaded yet)
  await sleep(500)
  const air = new Set(['air', 'cave_air', 'void_air'].flatMap(n => {
    const b = bot.registry.blocksByName[n]
    return Array.from({ length: b.maxStateId - b.minStateId + 1 }, (_, k) => b.minStateId + k)
  }))
  const blocks = []
  const v = new Vec3(0, 0, 0)
  for (let x = box[0]; x <= box[3]; x++) {
    for (let y = box[1]; y <= box[4]; y++) {
      for (let z = box[2]; z <= box[5]; z++) {
        v.x = x; v.y = y; v.z = z
        const id = bot.world.getBlockStateId(v) // (cheap; the Block object only for what is not air)
        if (id === null || id === undefined || air.has(id)) continue
        const b = bot.blockAt(v.clone(), false)
        if (b) blocks.push([x, y, z, b.name, b.getProperties(), b.shapes])
      }
    }
  }
  return { blocks, box }
}

async function goHome () {
  if (bot.entity.position.distanceTo(HOME) > 2) {
    await command(`/tp ${args.username} ${HOME.x} ${HOME.y} ${HOME.z}`)
    await waitFor(() => bot.entity.position.distanceTo(HOME) < 1, 10000, 'the teleport to the spawn')
  }
}

// The marker blocks within MARKER_RANGE of the spawn pad: { start: [[x, y, z], ...], goal: [...], via: [...] }.
async function findMarkers () {
  await goHome()
  const cols = [] // every chunk column with a point within MARKER_RANGE of HOME (the centre the search keeps)
  for (let cx = Math.floor((HOME.x - MARKER_RANGE) / 16); cx <= Math.floor((HOME.x + MARKER_RANGE) / 16); cx++) {
    for (let cz = Math.floor((HOME.z - MARKER_RANGE) / 16); cz <= Math.floor((HOME.z + MARKER_RANGE) / 16); cz++) {
      const dx = Math.max(cx * 16 - HOME.x, 0, HOME.x - cx * 16 - 16); const dz = Math.max(cz * 16 - HOME.z, 0, HOME.z - cz * 16 - 16)
      if (dx * dx + dz * dz <= MARKER_RANGE * MARKER_RANGE) cols.push([cx, cz])
    }
  }
  await waitFor(() => cols.every(([cx, cz]) => !!bot.world.getColumn(cx, cz)), 20000, 'the chunks around the spawn')
  const found = { start: [], goal: [], via: [] }
  for (const [name, role] of Object.entries(MARKERS)) { // (findBlocks walks an octahedron of chunk sections, not a
    // sphere: search twice as far and keep the sphere)
    found[role] = bot.findBlocks({ point: HOME, matching: bot.registry.blocksByName[name].id, maxDistance: 2 * MARKER_RANGE + 8, count: 256 })
      .filter(v => v.distanceTo(HOME) <= MARKER_RANGE).map(v => [v.x, v.y, v.z])
  }
  return found
}

async function convert () {
  if (!marks.source) { // no !start / !goal / !demo marks: the marker blocks
    const m = await findMarkers()
    if (m.start.length !== 1 || m.goal.length !== 1) {
      const at = ps => ps.slice(0, 4).map(p => p.join(' ')).join(', ')
      if (!m.start.length && !m.goal.length) {
        say(`mark the course first: a gold block as the start and a diamond block as the goal, within ${MARKER_RANGE} blocks of the spawn (or stand on them and say !start / !goal)`)
      }
      if (m.start.length > 1) say(`${m.start.length} gold blocks (start markers), at ${at(m.start)}: keep one`)
      if (m.goal.length > 1) say(`${m.goal.length} diamond blocks (goal markers), at ${at(m.goal)}: keep one`)
      if (!m.start.length && m.goal.length) say('no gold block (the start marker) yet')
      if (m.start.length && !m.goal.length) say('no diamond block (the goal marker) yet')
      return null
    }
    const [start] = m.start
    const d = p => Math.hypot(p[0] - start[0], p[1] - start[1], p[2] - start[2])
    marks.start = start
    marks.goal = m.goal[0]
    marks.via = m.via.sort((p, q) => d(p) - d(q))
    expect = null
    say(`markers: start ${marks.start.join(' ')}, goal ${marks.goal.join(' ')}` + (marks.via.length ? `, ${marks.via.length} via` : ''))
  }
  if (!marks.start || !marks.goal) {
    say('mark the start and the goal first: stand on the first block and say !start, on the last and say !goal (or !clear, then a gold and a diamond block)')
    return null
  }
  const { blocks, box } = await scan()
  const r = await converter.ask({ blocks, box, start: marks.start, goal: marks.goal, via: marks.via, name: `build ${runs + 1}`,
    ordered: marks.source === 'chat', ...(expect ? { expect } : {}) }) // (emerald markers count in any order)
  if (r.error) {
    say(`cannot play this build: ${r.error}`)
    for (const [x, y, z, name, why] of (r.refused || []).slice(0, 5)) say(`  ${name} at ${x} ${y} ${z}: ${why}`, false)
    return null
  }
  for (const w of r.warnings) say(`note: ${w}`, false)
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
  marks.source = 'demo'
  expect = { index: k, course: c, origin: DEMO_ORIGIN }
  await waitFor(() => !!bot.blockAt(new Vec3(...marks.start), false) && bot.blockAt(new Vec3(...marks.start), false).name !== 'air', 10000, 'the demo blocks')
  say(`demo course ${k} built at ${X} ${Y} ${Z} (${c.platforms.length - 1} jumps); start and goal marked: Run (or !run)`)
}

function describe (r) {
  const a = r.about
  const bits = [`${a.jumps} jumps`, `${a.neos.length} neo${a.neos.length === 1 ? '' : 's'}`]
  for (const k of ['ice', 'slime', 'shapes', 'ladders']) if (a[k]) bits.push(`${a[k]} ${k === 'shapes' ? 'shaped' : k}`)
  return bits.join(', ')
}

// The run; player (if any) watches it in spectator mode and is put back where they were, whatever happens (a
// spectator cannot press the panel's buttons). Marker blocks set on top of platforms (r.lifted) are off meanwhile.
async function playCourse (r, mode, player) {
  const c = r.course;
  [OX, OY, OZ] = r.origin
  const [sx, sy, sz] = c.spawn
  const coord = v => v.toFixed(6) // decimals: Minecraft centres whole-number /tp coordinates on the block
  const viewer = player && bot.players[player] ? player : null
  const seen = viewer && bot.players[viewer].entity // (out of the bot's sight, the viewer goes back to the spawn)
  const back = viewer ? (seen ? seen.position.clone() : HOME.clone()) : null
  const lifted = r.lifted || []
  const isAir = ([x, y, z]) => { const b = bot.blockAt(new Vec3(x, y, z), false); return !!b && b.name === 'air' }
  let ep = null
  let t0 = 0
  try {
    for (const [x, y, z] of lifted) await command(`/setblock ${x} ${y} ${z} minecraft:air`)
    if (viewer) await command(`/gamemode spectator ${viewer}`)
    await command(`/tp ${args.username} ${coord(sx + OX)} ${coord(sy + OY)} ${coord(sz + OZ)} ${toNotchianYawDeg(c.spawn_yaw).toFixed(4)} 0`)
    await waitFor(() => {
      const p = bot.entity.position
      return Math.abs(p.x - (sx + OX)) < 1e-6 && Math.abs(p.z - (sz + OZ)) < 1e-6 && Math.abs(p.y - (sy + OY)) < 1e-6
    }, 10000, 'the teleport')
    await waitFor(() => c.platforms.every(([x0, x1, z0, z1, y]) => isSolid(x0, y, z0) && isSolid(x1, y, z1)) && lifted.every(isAir),
      10000, 'the course blocks')
    if (viewer) await command(`/spectate ${args.username} ${viewer}`)
    await sleep(500) // a few ticks to settle on the start block, as the live bot does
    bot.entity.yaw = c.spawn_yaw
    corrections = 0
    lastCp = -1
    skillOn = false
    watcher = viewer
    const planner = mode === 'router' ? routerPlanner : null
    ep = new P.Episode(policy, c.platforms, isSolid, {
      stall: args.stall,
      maxTicks: args.maxTicks,
      extra: c.extra || [],
      planner,
      fallRule: args.fallRule,
      shapes: c.shapes || null,
      topOf,
      skill: skill ? { policy: skill, cps: c.skill_cps } : null
    })
    t0 = Date.now()
    await new Promise(resolve => { finish = resolve; episode = ep })
  } finally {
    episode = null
    watcher = null
    for (const [x, y, z, name] of lifted) await command(`/setblock ${x} ${y} ${z} minecraft:${name}`)
    if (viewer) {
      await command(`/gamemode creative ${viewer}`) // (which ends the spectating)
      if (back) await command(`/tp ${viewer} ${coord(back.x)} ${coord(back.y)} ${coord(back.z)}`)
    }
  }
  const rec = ep.record({ index: 0, mode, wall_seconds: (Date.now() - t0) / 1000, server_corrections: corrections })
  runsLog.write(JSON.stringify({ run: runs, course: c, origin: r.origin, mode, stopped: !!ep.stoppedByUser, episode: rec }) + '\n')
  runs++
  const n = c.platforms.length - 1
  if (ep.result === 'success') say(`made it: ${n} jumps in ${ep.ticks} ticks (${(ep.ticks / 20).toFixed(1)} s), ${corrections} server corrections`)
  else if (ep.result === 'fell') say(`fell after platform ${ep.cp} of ${n}${r.about.neos.includes(ep.cp) ? ' (at a neo)' : ''}`)
  else if (ep.stoppedByUser) say(`stopped (!stop) at platform ${ep.cp} of ${n}`)
  else say(`stopped at platform ${ep.cp} of ${n}: no progress for ${args.stall} ticks, or out of time`)
  if (!rec.actions || !rec.actions.length) { say('no ticks played, so nothing for the simulator to replay', false); return }
  const check = await converter.ask({ check: { course: c, episode: rec, mode } })
  if (check.error) say(`simulator check failed: ${check.error}`, false)
  else {
    say(`simulator replay of the game's ticks: largest difference ${check.max_pos_err.toExponential(1)} blocks; ` +
      `the simulator's own run: ${check.sim_success ? 'made it' : 'did not make it'} in ${check.sim_ticks} ticks` +
      (ep.stoppedByUser ? ' (the game run was stopped)' : check.sim_success === rec.success && check.sim_ticks === rec.ticks ? ' (the same)' : ' (different)'), false)
  }
}

const HELP = ['The panel by the spawn, chat (!build, ...) and the builder console (build, ...) give the same commands:',
  'Kit / !build: creative mode and the blocks the simulator models',
  'a gold block on (or in) the start platform marks it, a diamond block the goal, emerald blocks platforms the route must pass in any order (or stand on a block and say !start, !goal, !via)',
  'Route / !route: what the bot makes of the build. Run / !run [router|list]: the bot plays it, you watch',
  'Demo / !demo [K]: a ready-made course. Clear / !clear: forget !start, !goal and the demo (the markers count again). ' +
    'A run ends by itself; stop in the builder window (or !stop) ends it sooner']
const WELCOME = 'course builder: press Kit on the panel by the spawn (no chat needed), build, put a gold block on the start ' +
  'platform and a diamond block on the goal platform, then press Run. Help explains the rest.'
const SHORT = 'Kit, build, gold block on the start, diamond on the goal, then Run (chat set to Shown shows more)'

async function onCommand (player, line) {
  const [cmd, opt] = line.trim().split(/\s+/)
  if (cmd === '!help') { for (const h of HELP) say(h, false); bar(SHORT); return }
  if (cmd === '!stop') { if (episode) { episode.result = 'truncated'; episode.stoppedByUser = true; endEpisode() } return }
  if (busy) { say('busy: a run ends by itself (or type stop in the builder window, or !stop in chat)'); return }
  if (cmd === '!build') {
    if (!player) { say('Kit: no player to give it to'); return }
    await command(`/gamemode creative ${player}`)
    await command(`/clear ${player}`)
    for (const k of KIT) await command(`/give ${player} minecraft:${k} 64`)
    say('the kit: only blocks the simulator models. Platforms one block thick, a block of air between them, ' +
      'four blocks of air above. Walls of stone. A gold block on the start platform marks it, a diamond block the goal ' +
      '(emerald: a platform the route must pass). Then Route or Run')
    return
  }
  if (cmd === '!start' || cmd === '!goal' || cmd === '!via') {
    if (!player || !bot.players[player] || !bot.players[player].entity) { say(`${cmd}: the bot cannot see ${player || 'a player'} (come nearer to it)`); return }
    const b = blockUnder(player)
    if (!b) { say('stand on a block first (near its middle)'); return }
    if (marks.source !== 'chat') { marks.start = null; marks.goal = null; marks.via = [] }
    if (cmd === '!via') marks.via.push(b); else marks[cmd.slice(1)] = b
    marks.source = 'chat'
    expect = null
    say(`${cmd.slice(1)} at ${b.join(' ')}${cmd === '!via' ? ` (via ${marks.via.length})` : ''}`)
    return
  }
  if (cmd === '!clear') {
    marks.start = null; marks.goal = null; marks.via = []; marks.source = null; expect = null
    say('marks cleared: the marker blocks count again')
    return
  }
  if (cmd === '!demo') {
    const k = opt === undefined || opt === '' ? nextDemo : Number(opt)
    if (!Number.isInteger(k) || k < 0 || k >= demos.length) { say(`no demo course ${opt} (0 to ${demos.length - 1})`); return }
    busy = true
    try {
      await buildDemo(k)
      nextDemo = (k + 1) % demos.length
    } catch (err) { say(`error: ${err.message}`) } finally {
      await goHome().catch(err => log(err.message)) // (not left falling in the void by the demo area)
      busy = false
    }
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
          // (force: seen up to 512 blocks away, from the panel too; normal particles only within 32)
          if (k < 16) setTimeout(() => command(`/particle minecraft:happy_villager ${(x0 + x1 + 1) / 2 + r.origin[0]} ${y + 1.5 + r.origin[1]} ${(z0 + z1 + 1) / 2 + r.origin[2]} 0.3 0.3 0.3 0 15 force`), 400 * k)
        })
        return
      }
      const routerNo = !routerPlanner ? 'no router loaded' : r.about.router_ok ? '' : r.about.router_why
      const mode = opt === 'list' ? 'list' : opt === 'router' ? 'router' : (routerNo ? 'list' : 'router')
      if (mode === 'router' && routerNo) { say(`the learned router cannot play this build (${routerNo}): use Run (course list) or !run list`); return }
      say(mode === 'router' ? 'playing with the learned router (no course list), the final policy and the neo skill'
        : 'playing with the course list, the final policy and the neo skill' + (routerNo && opt !== 'list' ? ` (not the router: ${routerNo})` : ''))
      await playCourse(r, mode, player)
    } catch (err) {
      say(`error: ${err.message}`)
    } finally {
      await goHome().catch(err => log(err.message))
      busy = false
    }
  }
}

function run (player, line) {
  onCommand(player, line).catch(err => log('error: ' + (err && err.stack ? err.stack : err)))
}

bot.on('chat', (username, message) => {
  if (username === args.username || !message.startsWith('!')) return
  run(username, message)
})

// The panel's buttons: system messages from its command blocks (bridge/builder_panel.js).
bot.on('messagestr', (message, position) => {
  if (position === 'chat') return
  const c = Panel.parse(message)
  if (!c) return
  if (c.players.length > 1) { say(`several players by the panel (${c.players.join(', ')}): one presses, the others stand back`); return }
  run(c.players[0] || consolePlayer(), c.opt ? `${c.cmd} ${c.opt}` : c.cmd)
})

let panelBuilding = false // the bot's own panel commands are not damage
let repairing = null
async function buildPanel () {
  panelBuilding = true
  try {
    for (const c of Panel.commands(args.username)) await command(c)
    await sleep(2000)
  } finally { panelBuilding = false }
}

// A line typed in this console: the same commands without the "!" (for --viewer, or the one player online).
function consolePlayer () {
  const online = Object.keys(bot.players).filter(n => n !== args.username)
  if (args.viewer && online.includes(args.viewer)) return args.viewer
  return online.length === 1 ? online[0] : null
}
readline.createInterface({ input: process.stdin }).on('line', line => {
  const text = line.trim().replace(/^!/, '')
  if (text) run(consolePlayer(), '!' + text)
})

// A broken panel button or sign (one click in creative mode) is put back; a marker block placed means: take the
// marks from the markers again (not the bot's own lifting during a run).
bot.on('blockUpdate', (oldBlock, newBlock) => {
  if (!newBlock) return
  const p = newBlock.position
  const what = PANEL.get(`${p.x},${p.y},${p.z}`)
  if (what) {
    if (!panelBuilding && !repairing && newBlock.name !== PANEL_BLOCK[what]) {
      repairing = setTimeout(() => {
        repairing = null
        say('a panel block was broken: putting the panel back', false)
        buildPanel().catch(err => log(err.message))
      }, 1000)
    }
    return
  }
  const role = MARKERS[newBlock.name]
  if (busy || !role || (oldBlock && oldBlock.name === newBlock.name)) return
  if (marks.source) { marks.source = null; expect = null }
  bar(`${role} marker at ${p.x} ${p.y} ${p.z}`)
})

bot.on('playerJoined', player => {
  if (player.username === args.username) return
  setTimeout(() => { tell(WELCOME, player.username); bar(SHORT, player.username) }, 3000)
})

bot.once('spawn', async () => {
  log(`connected on port ${args.port}`)
  await sleep(1500)
  await buildPanel()
  say('builder ready: the panel by the spawn, chat (!help) or this console (help)')
})

bot.on('kicked', reason => { log('kicked: ' + JSON.stringify(reason)); converter.close(); process.exit(3) })
bot.on('error', err => { log('connection error: ' + err.message) })
bot.on('end', () => { converter.close(); process.exit(0) })
