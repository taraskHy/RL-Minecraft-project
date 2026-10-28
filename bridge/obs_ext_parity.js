'use strict'
// Extended-observation parity (tests/test_extended_obs.py): plays scripted actions on each course
// with prismarine-physics through policy.js's Episode, and writes every observation it builds.
//
//   node obs_ext_parity.js --cases cases.json --out obs.json

const fs = require('fs')
require('./physics_patch')
const { Vec3 } = require('vec3')
const { Physics } = require('prismarine-physics')
const mcData = require('minecraft-data')('1.21.4')
const P = require('./policy')

const args = {}
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, '')] = process.argv[i + 1]
const { courses, actions, settle } = JSON.parse(fs.readFileSync(args.cases, 'utf8'))
const STONE = mcData.blocksByName.stone.id
const AIR = mcData.blocksByName.air.id

const out = courses.map((c, index) => {
  const solid = new Set()
  for (const [x0, x1, z0, z1, y] of c.platforms) for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) solid.add(`${x},${y},${z}`)
  for (const [x, y, z] of c.extra || []) solid.add(`${x},${y},${z}`)
  const isSolid = (x, y, z) => solid.has(`${x},${y},${z}`)
  const world = {
    getBlock (p) {
      const x = Math.floor(p.x); const y = Math.floor(p.y); const z = Math.floor(p.z)
      const s = isSolid(x, y, z)
      return { type: s ? STONE : AIR, shapes: s ? [[0, 0, 0, 1, 1, 1]] : [], boundingBox: s ? 'block' : 'empty', position: new Vec3(x, y, z), metadata: 0 }
    }
  }
  const physics = Physics(mcData, world)
  const e = {
    pos: new Vec3(...c.spawn), vel: new Vec3(0, 0, 0), onGround: false, isInWater: false, isInLava: false, isInWeb: false,
    isCollidedHorizontally: false, isCollidedVertically: false, elytraFlying: false, jumpTicks: 0, jumpQueued: false,
    fireworkRocketDuration: 0, attributes: {}, yaw: c.spawn_yaw, pitch: 0,
    control: { forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: false },
    jumpBoost: 0, speed: 0, slowness: 0, dolphinsGrace: 0, slowFalling: 0, levitation: 0, depthStrider: 0, elytraEquipped: false
  }
  for (let t = 0; t < settle; t++) physics.simulatePlayer(e, world)
  const seen = []
  const script = actions[index]
  const policy = { obsDim: P.OBS_DIM_EXT, act (obs) { seen.push(Array.from(obs)); return script[seen.length - 1] } }
  const ep = new P.Episode(policy, c.platforms, isSolid, { stall: 100000, maxTicks: 100000, extra: c.extra || [] })
  for (let t = 0; t < script.length; t++) {
    const s = { pos: [e.pos.x, e.pos.y, e.pos.z], vel: [e.vel.x, e.vel.y, e.vel.z], yaw: e.yaw, onGround: e.onGround, jumpTicks: e.jumpTicks, collidedH: e.isCollidedHorizontally }
    const a = ep.step(s)
    if (a === null) break
    const k = P.controls(a)
    e.yaw = P.wrapYaw(e.yaw + k.yawStep)
    e.control = { forward: k.forward, back: k.back, left: k.left, right: k.right, jump: k.jump, sprint: k.sprint, sneak: false }
    physics.simulatePlayer(e, world)
  }
  return seen
})
fs.writeFileSync(args.out, JSON.stringify(out))
