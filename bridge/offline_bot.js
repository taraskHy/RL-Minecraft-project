'use strict'
// The live bot's episode loop without a server: the same policy.js Episode
// drives prismarine-physics (Mineflayer's own engine) on a world holding only
// the course blocks. It checks the whole JavaScript side end to end.
//
//   node offline_bot.js --policy P.json --courses C.json --out episodes.jsonl [--settle 10]
//                       [--planner '{"radius": 12}'] [--fall-rule local|remaining]
//                       [--assist CONFIG_JSON --assist-python PYTHON --assist-checkpoint CKPT --assist-root DIR --assist-device cpu|cuda]
//
// With --assist (bridge/assist.js) the courses carry "assist_takeoffs" and a Python planner
// service plans the bot's neos; every plan is checked with prismarine-physics before it is played.

const fs = require('fs')
const path = require('path')
require('./physics_patch') // vanilla collision tolerance, as in the simulator
const { Vec3 } = require('vec3')
const { Physics } = require('prismarine-physics')
const mcData = require('minecraft-data')('1.21.4')
const P = require('./policy')
const A = require('./assist')
// Shaped platforms are the real block states (prismarine-block is a dependency of Mineflayer).
const Block = require(require.resolve('prismarine-block', { paths: [path.dirname(require.resolve('mineflayer'))] }))('1.21.4')
const STATES = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'configs', 'block_shapes.json'), 'utf8'))
  .shapes.map(s => [s.name, s.state]))

const args = { settle: 10, stall: 200, maxTicks: 3000, planner: '', fallRule: 'local', assist: '' }
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = process.argv[i + 1]
}
const policy = P.Policy.load(args.policy)
const courses = JSON.parse(fs.readFileSync(args.courses, 'utf8')).courses
const out = fs.createWriteStream(args.out)
const STONE = mcData.blocksByName.stone.id
const ICE = mcData.blocksByName.ice.id
const SLIME = mcData.blocksByName.slime_block.id
const AIR = mcData.blocksByName.air.id

function makeWorld (platforms, extra, ice, shapes, slime, ladders) {
  const solid = new Set()
  const icy = new Set()
  const bouncy = new Set()
  const shaped = new Map()
  for (const [k, [x0, x1, z0, z1, y]] of platforms.entries()) {
    const isIce = (ice || []).includes(k)
    const isSlime = (slime || []).includes(k)
    const shape = shapes ? shapes[k] : undefined
    for (let x = x0; x <= x1; x++) {
      for (let z = z0; z <= z1; z++) {
        solid.add(`${x},${y},${z}`)
        if (isIce) icy.add(`${x},${y},${z}`)
        if (isSlime) bouncy.add(`${x},${y},${z}`)
        if (shape) shaped.set(`${x},${y},${z}`, Block.fromStateId(STATES[shape], 0))
      }
    }
  }
  for (const [x, y, z] of extra || []) solid.add(`${x},${y},${z}`)
  for (const [x, y, z, f] of ladders || []) { // ladders: solid thin boxes the physics climbs (by block type)
    solid.add(`${x},${y},${z}`)
    shaped.set(`${x},${y},${z}`, Block.fromStateId(STATES['ladder_' + ['north', 'east', 'south', 'west'][f]], 0))
  }
  const isSolid = (x, y, z) => solid.has(`${x},${y},${z}`)
  const topOf = (x, y, z) => { const b = shaped.get(`${x},${y},${z}`); return b ? Math.max(...b.shapes.map(q => q[4])) : 1 }
  const world = {
    getBlock (p) {
      const x = Math.floor(p.x); const y = Math.floor(p.y); const z = Math.floor(p.z)
      const real = shaped.get(`${x},${y},${z}`)
      if (real) return { type: real.type, shapes: real.shapes, boundingBox: real.boundingBox, position: new Vec3(x, y, z), metadata: real.metadata }
      const s = isSolid(x, y, z)
      const key = `${x},${y},${z}`
      const type = s ? (icy.has(key) ? ICE : bouncy.has(key) ? SLIME : STONE) : AIR
      return { type, shapes: s ? [[0, 0, 0, 1, 1, 1]] : [], boundingBox: s ? 'block' : 'empty', position: new Vec3(x, y, z), metadata: 0 }
    }
  }
  return { world, isSolid, topOf }
}

function entityAt (pos, yaw) {
  return {
    pos: new Vec3(pos[0], pos[1], pos[2]), vel: new Vec3(0, 0, 0), onGround: false, isInWater: false, isInLava: false,
    isInWeb: false, isCollidedHorizontally: false, isCollidedVertically: false, elytraFlying: false, jumpTicks: 0,
    jumpQueued: false, fireworkRocketDuration: 0, attributes: {}, yaw, pitch: 0,
    control: { forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: false },
    jumpBoost: 0, speed: 0, slowness: 0, dolphinsGrace: 0, slowFalling: 0, levitation: 0, depthStrider: 0, elytraEquipped: false
  }
}

const assistConfig = args.assist ? JSON.parse(args.assist) : null
const service = assistConfig
  ? new A.PlannerService(args.assistPython, A.serviceArgs(assistConfig, args.assistCheckpoint, args.courses, args.assistDevice), args.assistRoot)
  : null

async function main () {
  let ok = 0
  for (const [index, c] of courses.entries()) {
    const { world, isSolid, topOf } = makeWorld(c.platforms, c.extra, c.ice, c.shapes, c.slime, c.ladders)
    const physics = Physics(mcData, world)
    const e = entityAt(c.spawn, c.spawn_yaw)
    for (let t = 0; t < Number(args.settle); t++) physics.simulatePlayer(e, world)
    const ep = new P.Episode(policy, c.platforms, isSolid, {
      stall: Number(args.stall),
      maxTicks: Number(args.maxTicks),
      extra: c.extra || [],
      planner: args.planner ? JSON.parse(args.planner) : null,
      fallRule: args.fallRule,
      assist: assistConfig ? { config: assistConfig, takeoffs: c.assist_takeoffs || [], targets: c.assist_targets || null, retryHorizons: c.assist_retry_horizons || null, approach: c.assist_approach || null } : null,
      shapes: c.shapes || null,
      topOf
    })
    for (;;) {
      const s = { pos: [e.pos.x, e.pos.y, e.pos.z], vel: [e.vel.x, e.vel.y, e.vel.z], yaw: e.yaw, onGround: e.onGround, jumpTicks: e.jumpTicks, collidedH: e.isCollidedHorizontally }
      let a = ep.step(s)
      while (a === P.PLAN) { // a failed approach is followed by the takeoff's request in the same tick
        const q = ep.pending
        const r = await service.plan({ index, cp: q.cp, target: q.target, kind: q.kind, attempt: q.attempt, state: q.state, lastAction: q.lastAction })
        const verified = !!r.plan && A.verifyPlan(physics, world, q.state, r.plan, c.platforms, q.cp, q.target, c.extra, ep.geom)
        const brake = !r.plan && r.brake && A.verifyPlan(physics, world, q.state, r.brake, c.platforms, q.cp, q.cp, c.extra, ep.geom, true) ? r.brake : null
        a = ep.resume(verified ? r.plan : null, { found: !!r.plan, verified, rounds: r.rounds, seconds: r.seconds, error: r.error }, brake)
      }
      if (a === null) break
      const k = P.controls(a)
      e.yaw = P.wrapYaw(e.yaw + k.yawStep)
      e.control = { forward: k.forward, back: k.back, left: k.left, right: k.right, jump: k.jump, sprint: k.sprint, sneak: false }
      physics.simulatePlayer(e, world)
    }
    if (ep.result === 'success') ok++
    out.write(JSON.stringify(ep.record({ index, level: c.level, seed: c.seed, mode: 'offline' })) + '\n')
  }
  out.end()
  if (service) service.close()
  process.stdout.write(JSON.stringify({ courses: courses.length, success: ok }) + '\n')
}

main()
