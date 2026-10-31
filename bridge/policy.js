'use strict'
// Observation builder, episode rules and policy forward pass for the live bot.
// Mirrors parkour/env.py (Observer, standing_on, the fall and stall rules) and
// parkour/model.py (RunningNorm, actor MLP). tests/test_bridge_parity.py
// checks both against the Python code.

const fs = require('fs')
const path = require('path')
const { GreedyPlanner } = require('./planner')
const { LearnedRouter, RouterNet } = require('./router')

const routerNets = new Map() // loaded once per file
const { Assist, RouteAssist } = require('./assist')

const SCAN_FORWARD = Array.from({ length: 14 }, (_, i) => -1.0 + 0.5 * i)
const SCAN_LATERAL = Array.from({ length: 7 }, (_, i) => -1.5 + 0.5 * i)
const SCAN_FLOOR_LEVELS = [1, 0, -1, -2, -3, -4]
const SCAN_CEILING_LEVELS = [2, 3]
const NO_FLOOR = -5.0
const PLATFORM_SLOTS = 4
const STAND_EPS = 1e-6
const FALL_MARGIN = 1.0
const HALF_WIDTH = 0.3
const YAW_STEPS_DEG = [-20.0, -6.0, -1.5, 0.0, 1.5, 6.0, 20.0]
const ACTION_DIMS = [3, 3, 2, 2, 7]
const OBS_DIM = 6 + 5 + PLATFORM_SLOTS * 11 + SCAN_FORWARD.length * SCAN_LATERAL.length * 2 + 4
// Extended observation (env.OBS_LAYOUT_EXT): wide coarse scan and the last landing spots.
const SCAN_FAR_FORWARD = Array.from({ length: 15 }, (_, i) => -4.0 + i)
const SCAN_FAR_LATERAL = Array.from({ length: 13 }, (_, i) => -6.0 + i)
const MEMORY_SLOTS = 3
const OBS_DIM_EXT = OBS_DIM + SCAN_FAR_FORWARD.length * SCAN_FAR_LATERAL.length * 2 + MEMORY_SLOTS * 4
const NO_ACTION = [1, 1, 0, 0, 3]
const GRID = [64, 32, 64] // parkour.courses.GRID: the planner only looks inside the course area

function clamp (x, lo, hi) { return Math.min(hi, Math.max(lo, x)) }

function centre (p) { return [(p[0] + p[1] + 1) / 2, p[4] + 1, (p[2] + p[3] + 1) / 2] }

// Shaped platforms (Course.shapes, parkour/courses.shape_geometry): [top, x0, x1, z0, z1] of a block
// shape inside its cell, from configs/block_shapes.json.
let SHAPE_GEOMETRY = null
function shapeGeometry (name) {
  if (!SHAPE_GEOMETRY) {
    SHAPE_GEOMETRY = {}
    const file = path.join(__dirname, '..', 'configs', 'block_shapes.json')
    for (const sh of JSON.parse(fs.readFileSync(file, 'utf8')).shapes) {
      const b = sh.boxes
      SHAPE_GEOMETRY[sh.name] = [Math.max(...b.map(x => x[4])), Math.min(...b.map(x => x[0])), Math.max(...b.map(x => x[3])),
        Math.min(...b.map(x => x[2])), Math.max(...b.map(x => x[5]))]
    }
  }
  return SHAPE_GEOMETRY[name]
}

// Standing area and top of every platform, [x0, x1, z0, z1, top]; ``shapes`` ({index: shape name}) as in
// CoursePool.foot / CoursePool.top.
function platformGeometry (platforms, shapes) {
  return platforms.map((p, k) => {
    const name = shapes ? shapes[k] : undefined
    if (!name) return [p[0], p[1] + 1, p[2], p[3] + 1, p[4] + 1]
    const [top, x0, x1, z0, z1] = shapeGeometry(name)
    return [p[0] + x0, p[1] + x1, p[2] + z0, p[3] + z1, p[4] + top]
  })
}

// s: {pos:[x,y,z], vel:[x,y,z], yaw, onGround, jumpTicks, collidedH}
// platforms: [[x0,x1,z0,z1,y], ...] in the same coordinates as s.pos
// isSolid(x, y, z): integer cell -> boolean
// geom (shaped courses): platformGeometry(...); topOf(x, y, z): the top of a solid cell's shape (1 for full blocks)
function observe (s, lastAction, platforms, cp, isSolid, geom = null, topOf = null) {
  const n = platforms.length
  const slots = []
  for (let k = 0; k < PLATFORM_SLOTS && cp + k < n; k++) {
    slots.push({ p: platforms[cp + k], goal: cp + k === n - 1, g: geom ? geom[cp + k] : undefined })
  }
  const g = geom ? geom[n - 1] : null
  const goal = g ? [(g[0] + g[1]) / 2, g[4], (g[2] + g[3]) / 2] : centre(platforms[n - 1])
  return observeSlots(s, lastAction, slots, goal, cp / Math.max(n - 1, 1), isSolid, null, geom ? topOf : null)
}

// The observation with explicit platform slots ([{p, goal}], current platform first; missing
// slots are empty), goal centre and progress: what observe() builds from the course, or what
// the route planner builds from the blocks it sees.
function observeSlots (s, lastAction, slots, g, progress, isSolid, memory = null, topOf = null) {
  const out = new Float64Array(memory ? OBS_DIM_EXT : OBS_DIM)
  let o = 0
  const [px, py, pz] = s.pos
  const sy = Math.sin(s.yaw)
  const cy = Math.cos(s.yaw)
  const fx = -sy; const fz = -cy
  const rx = cy; const rz = -sy
  const relF = (x, z) => (x - px) * fx + (z - pz) * fz
  const relR = (x, z) => (x - px) * rx + (z - pz) * rz
  const v = s.vel

  out[o++] = (v[0] * fx + v[2] * fz) / 0.4
  out[o++] = (v[0] * rx + v[2] * rz) / 0.4
  out[o++] = v[1] / 0.5
  out[o++] = s.onGround ? 1 : 0
  out[o++] = s.jumpTicks / 10
  out[o++] = s.collidedH ? 1 : 0

  out[o++] = lastAction[0] - 1
  out[o++] = lastAction[1] - 1
  out[o++] = lastAction[2]
  out[o++] = lastAction[3]
  out[o++] = YAW_STEPS_DEG[lastAction[4]] / 20

  for (let k = 0; k < PLATFORM_SLOTS; k++) {
    if (k >= slots.length) { o += 11; continue }
    const p = slots[k].p
    const q = slots[k].g // a shaped platform's own footprint and top
    const x0 = q ? q[0] : p[0]; const x1 = q ? q[1] : p[1] + 1; const z0 = q ? q[2] : p[2]; const z1 = q ? q[3] : p[3] + 1
    out[o++] = 1
    out[o++] = slots[k].goal ? 1 : 0
    const cxs = [x0, x1, x1, x0]
    const czs = [z0, z0, z1, z1]
    for (let c = 0; c < 4; c++) {
      out[o++] = clamp(relF(cxs[c], czs[c]) / 4.0, -10, 10)
      out[o++] = clamp(relR(cxs[c], czs[c]) / 4.0, -10, 10)
    }
    out[o++] = q ? (q[4] - py) / 2.0 : ((p[4] + 1) - py) / 2.0
  }

  const fy = Math.floor(py + STAND_EPS)
  const scan = (fwd, lat) => {
    for (const a of fwd) {
      for (const b of lat) {
        const sx = px + a * fx + b * rx
        const sz = pz + a * fz + b * rz
        const ix = Math.floor(sx)
        const iz = Math.floor(sz)
        let h = NO_FLOOR
        for (const lv of SCAN_FLOOR_LEVELS) {
          if (isSolid(ix, fy + lv, iz)) { h = topOf ? (fy + lv + topOf(ix, fy + lv, iz)) - py : (fy + lv + 1) - py; break }
        }
        let ceiling = 0
        for (const lv of SCAN_CEILING_LEVELS) {
          if (isSolid(ix, fy + lv, iz)) { ceiling = 1; break }
        }
        out[o++] = h / 3.0
        out[o++] = ceiling
      }
    }
  }
  scan(SCAN_FORWARD, SCAN_LATERAL)

  out[o++] = relF(g[0], g[2]) / 20
  out[o++] = relR(g[0], g[2]) / 20
  out[o++] = (g[1] - py) / 10
  out[o++] = progress
  if (memory) {
    scan(SCAN_FAR_FORWARD, SCAN_FAR_LATERAL)
    for (let k = 0; k < MEMORY_SLOTS; k++) {
      if (!memory.valid[k]) { o += 4; continue }
      const m = memory.pos[k]
      out[o++] = 1
      out[o++] = relF(m[0], m[2]) / 4.0
      out[o++] = relR(m[0], m[2]) / 4.0
      out[o++] = (m[1] - py) / 2.0
    }
  }
  return out
}

// Landing memory (EnvConfig.extended_obs): the last MEMORY_SLOTS spots landed on, newest first.
class LandingMemory {
  constructor (pos, onGround) {
    this.pos = [pos.slice(), [0, 0, 0], [0, 0, 0]]
    this.valid = [true, false, false]
    this.prevGround = onGround
  }

  update (s) {
    const dx = s.pos[0] - this.pos[0][0]; const dz = s.pos[2] - this.pos[0][2]
    if (s.onGround && !this.prevGround && (dx * dx + dz * dz >= 1.0 || !this.valid[0])) {
      this.pos = [s.pos.slice(), this.pos[0], this.pos[1]]
      this.valid = [true, this.valid[0], this.valid[1]]
    }
    this.prevGround = s.onGround
  }
}

// Highest platform index the player stands on, or -1 (geom: platformGeometry for shaped courses).
function standingOn (s, platforms, geom = null) {
  if (!s.onGround) return -1
  const [x, y, z] = s.pos
  let best = -1
  for (let k = 0; k < platforms.length; k++) {
    const p = platforms[k]
    if (geom) {
      const q = geom[k]
      const over = x + HALF_WIDTH > q[0] && x - HALF_WIDTH < q[1] && z + HALF_WIDTH > q[2] && z - HALF_WIDTH < q[3]
      if (over && Math.abs(y - q[4]) < STAND_EPS) best = k
      continue
    }
    const over = x + HALF_WIDTH > p[0] && x - HALF_WIDTH < p[1] + 1 && z + HALF_WIDTH > p[2] && z - HALF_WIDTH < p[3] + 1
    if (over && Math.abs(y - (p[4] + 1)) < STAND_EPS) best = k
  }
  return best
}

// rule 'local': below platforms cp-1..cp+2 (as in training); 'remaining': below every platform
// from cp-1 to the goal (EnvConfig.fall_rule).
function fellBelow (s, platforms, cp, rule = 'local', geom = null) {
  const last = platforms.length - 1
  let lo = Infinity
  for (let j = cp - 1; j <= (rule === 'local' ? cp + 2 : last); j++) {
    if (j >= 0 && j <= last) lo = Math.min(lo, geom ? geom[j][4] : platforms[j][4] + 1)
  }
  return s.pos[1] < lo - FALL_MARGIN
}

class Policy {
  constructor (data) {
    if (data.format !== 'parkour-policy-v1') throw new Error('unknown policy format ' + data.format)
    if (data.obs_dim !== OBS_DIM && data.obs_dim !== OBS_DIM_EXT) throw new Error(`policy expects ${data.obs_dim} observations, bridge builds ${OBS_DIM} or ${OBS_DIM_EXT}`)
    this.mean = data.norm_mean
    this.std = data.norm_std
    this.clip = data.norm_clip
    this.mask = data.obs_mask || null
    this.layers = data.layers
    this.actionDims = data.action_dims
    this.obsDim = data.obs_dim
    this.meta = data.meta
  }

  static load (path) { return new Policy(JSON.parse(fs.readFileSync(path, 'utf8'))) }

  logits (obs) {
    let x = new Float64Array(obs.length)
    for (let i = 0; i < obs.length; i++) {
      x[i] = clamp((obs[i] - this.mean[i]) / this.std[i], -this.clip, this.clip)
      if (this.mask) x[i] *= this.mask[i]
    }
    for (let l = 0; l < this.layers.length; l++) {
      const { w, b } = this.layers[l]
      const y = new Float64Array(b.length)
      for (let r = 0; r < b.length; r++) {
        const row = w[r]
        let acc = b[r]
        for (let c = 0; c < x.length; c++) acc += row[c] * x[c]
        y[r] = l < this.layers.length - 1 ? (acc > 0 ? acc : Math.expm1(acc)) : acc
      }
      x = y
    }
    return x
  }

  // Deterministic action: argmax per component (what evaluation uses).
  act (obs) {
    const lg = this.logits(obs)
    const action = []
    let o = 0
    for (const d of this.actionDims) {
      let best = 0
      for (let i = 1; i < d; i++) if (lg[o + i] > lg[o + best]) best = i
      action.push(best)
      o += d
    }
    return action
  }
}

// Action -> Mineflayer control states and the yaw step in radians.
function controls (action) {
  return {
    forward: action[0] === 2,
    back: action[0] === 0,
    left: action[1] === 0,
    right: action[1] === 2,
    jump: action[2] === 1,
    sprint: action[3] === 1,
    yawStep: YAW_STEPS_DEG[action[4]] * Math.PI / 180
  }
}

function wrapYaw (yaw) {
  // Same as torch.remainder(yaw + pi, 2 pi) - pi.
  const t = 2 * Math.PI
  let r = (yaw + Math.PI) % t
  if (r < 0) r += t
  return r - Math.PI
}

// One episode on one course, fed the player state after every game tick.
// step(s) returns the action for the next tick, or null once the episode has
// ended (this.result is then 'success', 'fell' or 'truncated'). The rules are
// those of ParkourEnv.step.
// True when standing on top of an obstacle cell (wall, beam, pillar): ends the episode like a fall.
function onObstacle (s, extra) {
  if (!s.onGround || !extra) return false
  const [x, y, z] = s.pos
  for (const [bx, by, bz] of extra) {
    if (Math.abs(y - (by + 1)) < STAND_EPS && x + HALF_WIDTH > bx && x - HALF_WIDTH < bx + 1 && z + HALF_WIDTH > bz && z - HALF_WIDTH < bz + 1) return true
  }
  return false
}

// Episode.step returns PLAN when the neo assist wants a plan (bridge/assist.js): the bot then gets
// one for episode.pending and calls episode.resume(plan or null) for the action.
const PLAN = 'plan'

class Episode {
  constructor (policy, platforms, isSolid, { stall = 200, maxTicks = 3000, extra = null, planner = null, fallRule = 'local', assist = null, shapes = null, topOf = null } = {}) {
    this.extra = extra
    // Shaped platforms ({index: shape name}): their own footprints and tops, and the shape tops in the scan.
    this.geom = shapes && Object.keys(shapes).length ? platformGeometry(platforms, shapes) : null
    this.topOf = topOf
    if (this.geom && planner) throw new Error('the route planner does not support shaped platforms')
    this.queue = []
    this.pending = null
    this.assistLog = []
    this.fallRule = fallRule
    // With a planner config, the policy's platform slots come from the blocks around the bot
    // (bridge/planner.js); the course list is only used to score the episode.
    this.planner = null
    if (planner) {
      const inGrid = (x, y, z) => x >= 0 && x < GRID[0] && y >= 0 && y < GRID[1] && z >= 0 && z < GRID[2] && isSolid(x, y, z)
      const goal = centre(platforms[platforms.length - 1])
      if (planner.router) { // a learned route policy (bridge/router.js) instead of the greedy rule
        if (!routerNets.has(planner.router)) routerNets.set(planner.router, RouterNet.load(planner.router))
        const { router, fill, ...cfg } = planner
        this.planner = new LearnedRouter(inGrid, goal, routerNets.get(router), cfg, fill || 'greedy')
      } else {
        this.planner = new GreedyPlanner(inGrid, goal, planner)
      }
    }
    // With an assist ({config, takeoffs, targets, retryHorizons, approach}), planned actions replace the policy's (see
    // PLAN); with the learned router and "blocked": "plan", its blocked-path choices are planned (RouteAssist).
    this.assist = null
    if (assist && planner && planner.router && planner.blocked === 'plan') {
      this.assist = new RouteAssist(assist.config, this.planner, platforms, this.geom)
    } else if (assist) {
      this.assist = new Assist(assist.config, assist.takeoffs, assist.targets, assist.retryHorizons, assist.approach)
    }
    this.policy = policy
    this.platforms = platforms
    this.isSolid = isSolid
    this.stallLimit = stall
    this.maxTicks = maxTicks
    this.cp = 0
    this.stall = 0
    this.ticks = 0
    this.started = false
    this.result = null
    this.lastAction = NO_ACTION.slice()
    this.extended = policy.obsDim === OBS_DIM_EXT
    this.memory = null
    this.actions = []
    this.traj = []
    this.init = null
  }

  step (s) {
    if (this.result) return null
    if (this.started) {
      this.ticks++
      this.traj.push([s.pos[0], s.pos[1], s.pos[2], s.vel[0], s.vel[1], s.vel[2], s.onGround ? 1 : 0, s.jumpTicks])
      const k = standingOn(s, this.platforms, this.geom)
      if (k > this.cp) { this.cp = k; this.stall = 0 } else this.stall++
      const success = this.cp >= this.platforms.length - 1
      const fell = !success && (fellBelow(s, this.platforms, this.cp, this.fallRule, this.geom) || onObstacle(s, this.extra))
      const truncated = !success && !fell && (this.stall >= this.stallLimit || this.ticks >= this.maxTicks)
      if (success || fell || truncated) {
        this.result = success ? 'success' : fell ? 'fell' : 'truncated'
        return null
      }
      if (this.planner) this.planner.update(s.pos, s.onGround, this.ticks, s)
      if (this.memory) this.memory.update(s)
      if (this.assist && this.queue.length === 0 && this._ask(s)) return PLAN
    } else {
      this.started = true
      this.init = s
      if (this.planner) this.planner.start(s.pos, s)
      if (this.extended) this.memory = new LandingMemory(s.pos, s.onGround)
      // plan_start: the start state is searched from too, before the first action (the bot settled on the start block)
      if (this.assist && this.assist.cfg.plan_start && this._ask(s)) return PLAN
    }
    return this._act(s)
  }

  // The assist's request for state s, if its rules make one: then this.pending is set and the result is true.
  _ask (s) {
    const req = this.assist.request(s, this.cp, this.stall, this.ticks, this.stallLimit)
    if (req) this.pending = { ...req, tick: this.ticks, state: s, lastAction: this.lastAction.slice() }
    return !!req
  }

  // After PLAN: play ``plan`` (a list of actions) or, with null, let the policy act; with no plan, a
  // ``brake`` (assist.js: keys that stop the bot on the takeoff) is played and the retry follows it. An
  // approach without a plan is followed by the takeoff's own request in the same tick (PLAN again).
  resume (plan, info = {}, brake = null) {
    const req = this.pending
    this.pending = null
    this.assistLog.push({ kind: req.kind, cp: req.cp, takeoff: req.takeoff, tick: req.tick, ...info, played: !!plan, braked: !plan && !!brake })
    if (plan) {
      this.queue = plan.map(a => a.slice())
      if (this.assist.played) this.assist.played(req, plan.length) // RouteAssist: plan_on follows the landing
    } else if (brake) {
      this.queue = brake.map(a => a.slice())
      this.assist.braked = true
    } else if (req.kind === 'approach') {
      const next = this.assist.request(req.state, this.cp, this.stall, this.ticks, this.stallLimit)
      if (next) {
        this.pending = { ...next, tick: this.ticks, state: req.state, lastAction: req.lastAction }
        return PLAN
      }
    }
    return this._act(req.state)
  }

  _act (s) {
    if (this.queue.length) {
      const a = this.queue.shift()
      this.actions.push(a)
      this.lastAction = a
      return a
    }
    const n = this.platforms.length
    let slots, progress
    if (this.planner) {
      slots = this.planner.slots()
      progress = this.planner.progress
    } else {
      slots = []
      for (let k = 0; k < PLATFORM_SLOTS && this.cp + k < n; k++) {
        slots.push({ p: this.platforms[this.cp + k], goal: this.cp + k === n - 1, g: this.geom ? this.geom[this.cp + k] : undefined })
      }
      progress = this.cp / Math.max(n - 1, 1)
    }
    const gq = this.geom ? this.geom[n - 1] : null
    const goal = gq ? [(gq[0] + gq[1]) / 2, gq[4], (gq[2] + gq[3]) / 2] : centre(this.platforms[n - 1])
    const obs = observeSlots(s, this.lastAction, slots, goal, progress, this.isSolid, this.memory, this.geom ? this.topOf : null)
    const a = this.policy.act(obs)
    this.actions.push(a)
    this.lastAction = a
    return a
  }

  record (extra) {
    return {
      ...extra,
      n: this.platforms.length,
      result: this.result,
      success: this.result === 'success',
      cp: this.cp,
      ticks: this.ticks,
      init: this.init,
      actions: this.actions,
      traj: this.traj,
      ...(this.planner ? { plans: this.planner.events } : {}),
      ...(this.assist ? { assist: this.assistLog } : {})
    }
  }
}

module.exports = {
  observe, observeSlots, LandingMemory, OBS_DIM_EXT, standingOn, fellBelow, onObstacle, Policy, Episode, controls, wrapYaw, centre,
  OBS_DIM, ACTION_DIMS, NO_ACTION, YAW_STEPS_DEG, PLAN, platformGeometry, shapeGeometry
}
