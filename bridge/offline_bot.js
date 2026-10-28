'use strict'
// The live bot's episode loop without a server: the same policy.js Episode
// drives prismarine-physics (Mineflayer's own engine) on a world holding only
// the course blocks. It checks the whole JavaScript side end to end.
//
//   node offline_bot.js --policy P.json --courses C.json --out episodes.jsonl [--settle 10]
//                       [--planner '{"radius": 12}'] [--fall-rule local|remaining]

const fs = require('fs')
require('./physics_patch') // vanilla collision tolerance, as in the simulator
const { Vec3 } = require('vec3')
const { Physics } = require('prismarine-physics')
const mcData = require('minecraft-data')('1.21.4')
const P = require('./policy')

const args = { settle: 10, stall: 200, maxTicks: 3000, planner: '', fallRule: 'local' }
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = process.argv[i + 1]
}
const policy = P.Policy.load(args.policy)
const courses = JSON.parse(fs.readFileSync(args.courses, 'utf8')).courses
const out = fs.createWriteStream(args.out)
const STONE = mcData.blocksByName.stone.id
const AIR = mcData.blocksByName.air.id

function makeWorld (platforms, extra) {
  const solid = new Set()
  for (const [x0, x1, z0, z1, y] of platforms) {
    for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) solid.add(`${x},${y},${z}`)
  }
  for (const [x, y, z] of extra || []) solid.add(`${x},${y},${z}`)
  const isSolid = (x, y, z) => solid.has(`${x},${y},${z}`)
  const world = {
    getBlock (p) {
      const x = Math.floor(p.x); const y = Math.floor(p.y); const z = Math.floor(p.z)
      const s = isSolid(x, y, z)
      return { type: s ? STONE : AIR, shapes: s ? [[0, 0, 0, 1, 1, 1]] : [], boundingBox: s ? 'block' : 'empty', position: new Vec3(x, y, z), metadata: 0 }
    }
  }
  return { world, isSolid }
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

let ok = 0
courses.forEach((c, index) => {
  const { world, isSolid } = makeWorld(c.platforms, c.extra)
  const physics = Physics(mcData, world)
  const e = entityAt(c.spawn, c.spawn_yaw)
  for (let t = 0; t < Number(args.settle); t++) physics.simulatePlayer(e, world)
  const ep = new P.Episode(policy, c.platforms, isSolid, {
    stall: Number(args.stall), maxTicks: Number(args.maxTicks), extra: c.extra || [], planner: args.planner ? JSON.parse(args.planner) : null, fallRule: args.fallRule
  })
  for (;;) {
    const s = { pos: [e.pos.x, e.pos.y, e.pos.z], vel: [e.vel.x, e.vel.y, e.vel.z], yaw: e.yaw, onGround: e.onGround, jumpTicks: e.jumpTicks, collidedH: e.isCollidedHorizontally }
    const a = ep.step(s)
    if (a === null) break
    const k = P.controls(a)
    e.yaw = P.wrapYaw(e.yaw + k.yawStep)
    e.control = { forward: k.forward, back: k.back, left: k.left, right: k.right, jump: k.jump, sprint: k.sprint, sneak: false }
    physics.simulatePlayer(e, world)
  }
  if (ep.result === 'success') ok++
  out.write(JSON.stringify(ep.record({ index, level: c.level, seed: c.seed, mode: 'offline' })) + '\n')
})
out.end()
process.stdout.write(JSON.stringify({ courses: courses.length, success: ok }) + '\n')
