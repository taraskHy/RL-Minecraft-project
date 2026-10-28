'use strict'
// Runs the real prismarine-physics engine (the one Mineflayer uses) on given
// worlds and control sequences, and prints the trajectories as JSON.
//
// Input (file path argument, JSON):
//   { "cases": [ { "blocks": [[x,y,z], ...], "ice": [[x,y,z], ...] (optional: which of the blocks are ice),
//                  "shapes": [[x,y,z,name], ...] (optional: blocks that are the block state of that name
//                  in ../configs/block_shapes.json; their collision boxes come from prismarine-block),
//                  "pos": [x,y,z], "vel": [x,y,z],
//                  "onGround": false, "jumpTicks": 0,
//                  "ticks": [ { "yaw": 0.0, "c": [fwd, back, left, right, jump, sprint] }, ... ] } ] }
// Output (stdout, JSON): { "cases": [ [ {pos, vel, onGround, jumpTicks, collH, collV}, ... ], ... ] }

const fs = require('fs')
const path = require('path')
require('./physics_patch') // vanilla collision tolerance, as in the simulator
const { Vec3 } = require('vec3')
const { Physics } = require('prismarine-physics')
const mcData = require('minecraft-data')('1.21.4')
const Block = require(require.resolve('prismarine-block', { paths: [path.dirname(require.resolve('mineflayer'))] }))('1.21.4')
const STATES = Object.fromEntries(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'configs', 'block_shapes.json'), 'utf8'))
  .shapes.map(s => [s.name, s.state]))

const STONE = mcData.blocksByName.stone.id
const ICE = mcData.blocksByName.ice.id
const AIR = mcData.blocksByName.air.id

function makeWorld (blocks, ice, shapes) {
  const solid = new Set(blocks.map(b => `${b[0]},${b[1]},${b[2]}`))
  const icy = new Set((ice || []).map(b => `${b[0]},${b[1]},${b[2]}`))
  const shaped = new Map((shapes || []).map(b => [`${b[0]},${b[1]},${b[2]}`, Block.fromStateId(STATES[b[3]], 0)]))
  return {
    getBlock (p) {
      const x = Math.floor(p.x)
      const y = Math.floor(p.y)
      const z = Math.floor(p.z)
      const key = `${x},${y},${z}`
      const isSolid = solid.has(key)
      const real = shaped.get(key)
      if (real) {
        return { type: real.type, shapes: real.shapes, boundingBox: real.boundingBox, position: new Vec3(x, y, z), metadata: real.metadata,
          isWaterlogged: false, getProperties () { return real.getProperties() } }
      }
      return {
        type: isSolid ? (icy.has(key) ? ICE : STONE) : AIR,
        shapes: isSolid ? [[0, 0, 0, 1, 1, 1]] : [],
        boundingBox: isSolid ? 'block' : 'empty',
        position: new Vec3(x, y, z),
        metadata: 0,
        isWaterlogged: false,
        getProperties () { return {} }
      }
    }
  }
}

function makeEntity (c) {
  return {
    pos: new Vec3(c.pos[0], c.pos[1], c.pos[2]),
    vel: new Vec3(c.vel[0], c.vel[1], c.vel[2]),
    onGround: !!c.onGround,
    isInWater: false,
    isInLava: false,
    isInWeb: false,
    isCollidedHorizontally: false,
    isCollidedVertically: false,
    elytraFlying: false,
    jumpTicks: c.jumpTicks || 0,
    jumpQueued: false,
    fireworkRocketDuration: 0,
    attributes: {},
    yaw: 0,
    pitch: 0,
    control: { forward: false, back: false, left: false, right: false, jump: false, sprint: false, sneak: false },
    jumpBoost: 0,
    speed: 0,
    slowness: 0,
    dolphinsGrace: 0,
    slowFalling: 0,
    levitation: 0,
    depthStrider: 0,
    elytraEquipped: false
  }
}

function runCase (c) {
  const world = makeWorld(c.blocks, c.ice, c.shapes)
  const physics = Physics(mcData, world)
  const e = makeEntity(c)
  const out = []
  for (const t of c.ticks) {
    e.yaw = t.yaw
    const k = t.c
    e.control = { forward: !!k[0], back: !!k[1], left: !!k[2], right: !!k[3], jump: !!k[4], sprint: !!k[5], sneak: false }
    physics.simulatePlayer(e, world)
    out.push({
      pos: [e.pos.x, e.pos.y, e.pos.z],
      vel: [e.vel.x, e.vel.y, e.vel.z],
      onGround: e.onGround,
      jumpTicks: e.jumpTicks,
      collH: e.isCollidedHorizontally,
      collV: e.isCollidedVertically
    })
  }
  return out
}

const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
process.stdout.write(JSON.stringify({ cases: input.cases.map(runCase) }))
