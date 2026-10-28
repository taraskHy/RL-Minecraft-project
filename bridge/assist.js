'use strict'
// Neo assist for the bots (parkour/neo_assist.py). The first time the bot stands on a planned
// takeoff (a neo's, or with "jumps": "advanced" also a wide or momentum jump's), the episode asks
// for a plan; the bot pauses, the Python planner service searches an action sequence from the
// bot's exact state with the simulator, the bot checks the sequence with prismarine-physics on its
// own world, and plays it; then the policy plays on. Assist mirrors the trigger rules of
// evaluate_assisted (tests/test_neo_assist.py checks the offline bot against the simulator).

const { spawn } = require('child_process')
const readline = require('readline')
const { Vec3 } = require('vec3')

const HALF_WIDTH = 0.3
const STAND_EPS = 1e-6
const YAW_STEPS_DEG = [-20.0, -6.0, -1.5, 0.0, 1.5, 6.0, 20.0]

// AssistConfig's defaults (parkour/neo_assist.py) for the fields the trigger rules read, so a partial
// --assist JSON behaves as in Python (tests/test_neo_assist.py checks they agree).
const DEFAULTS = { horizon: 32, retry_horizon: 0, retries: 0, retry_every: 10, approach: 0 }

// When to plan, per episode: AssistConfig fields (snake_case, as parkour/neo_assist.py writes them).
class Assist {
  constructor (cfg, takeoffs) {
    cfg = { ...DEFAULTS, ...cfg }
    this.cfg = cfg
    this.planned = new Set(takeoffs)
    this.before = new Set(takeoffs.filter(k => k >= 1 && !this.planned.has(k - 1)).map(k => k - 1))
    this.tries = new Map()
    this.triedBefore = new Set()
    this.lastTry = -1e9
    this.retryHorizon = cfg.retry_horizon || cfg.horizon
  }

  // The request after this tick's rules, or null (no plan may be playing).
  request (s, cp, stall, ticks, stallLimit) {
    if (!s.onGround) return null
    const n = this.tries.get(cp) || 0
    if (this.planned.has(cp)) {
      if (n === 0) return this._take('takeoff', cp, cp + 1, 0, ticks)
      if (n <= this.cfg.retries && ticks - this.lastTry >= this.cfg.retry_every && stall + this.retryHorizon < stallLimit) {
        return this._take('retry', cp, cp + 1, n, ticks)
      }
      return null
    }
    if (this.cfg.approach > 0 && this.before.has(cp) && !this.triedBefore.has(cp)) return this._take('approach', cp, cp + 2, 0, ticks)
    return null
  }

  _take (kind, cp, target, attempt, ticks) {
    if (kind === 'approach') {
      this.triedBefore.add(cp)
    } else {
      this.tries.set(cp, (this.tries.get(cp) || 0) + 1)
      this.lastTry = ticks
    }
    return { kind, cp, target, attempt, takeoff: kind === 'approach' ? cp + 1 : cp }
  }
}

// The Python planner (python -m parkour.neo_assist --serve), one JSON line per request and answer.
class PlannerService {
  constructor (python, args, cwd) {
    this.proc = spawn(python, ['-m', 'parkour.neo_assist', '--serve', ...args], { cwd, stdio: ['pipe', 'pipe', 'inherit'] })
    this.waiting = []
    this.dead = null // why the planner can no longer answer; requests then get no plan at once
    readline.createInterface({ input: this.proc.stdout }).on('line', line => {
      const w = this.waiting.shift()
      if (w) w(JSON.parse(line))
    })
    const die = why => {
      this.dead = this.dead || why
      for (const w of this.waiting.splice(0)) w({ plan: null, error: this.dead })
    }
    this.proc.on('exit', code => die(`planner exited (${code})`))
    this.proc.on('error', err => die(`planner failed: ${err.message}`))
    this.proc.stdin.on('error', err => die(`planner input closed: ${err.message}`))
  }

  plan (req) {
    if (this.dead) return Promise.resolve({ plan: null, error: this.dead })
    return new Promise(resolve => {
      this.waiting.push(resolve)
      this.proc.stdin.write(JSON.stringify(req) + '\n')
    })
  }

  close () { this.proc.stdin.end() }
}

// Command-line flags of the planner service for an AssistConfig (device: that of the simulator it is compared with).
function serviceArgs (cfg, checkpoint, coursesFile, device = '') {
  const out = ['--checkpoint', checkpoint, '--courses-file', coursesFile]
  if (device) out.push('--device', device)
  for (const [k, v] of Object.entries(cfg)) if (k !== 'enabled') out.push('--' + k.replace(/_/g, '-'), String(v))
  return out
}

function standsOn (s, p) {
  const [x, y, z] = s.pos
  return s.onGround && x + HALF_WIDTH > p[0] && x - HALF_WIDTH < p[1] + 1 && z + HALF_WIDTH > p[2] && z - HALF_WIDTH < p[3] + 1 &&
    Math.abs(y - (p[4] + 1)) < STAND_EPS
}

function onObstacle (s, extra) {
  if (!s.onGround) return false
  const [x, y, z] = s.pos
  for (const [bx, by, bz] of extra || []) {
    if (Math.abs(y - (by + 1)) < STAND_EPS && x + HALF_WIDTH > bx && x - HALF_WIDTH < bx + 1 && z + HALF_WIDTH > bz && z - HALF_WIDTH < bz + 1) return true
  }
  return false
}

function wrapYaw (yaw) {
  const t = 2 * Math.PI
  let r = (yaw + Math.PI) % t
  if (r < 0) r += t
  return r - Math.PI
}

// Play a plan on a copy of the bot with prismarine-physics on the bot's own world: it must stay off
// obstacles and above the lowest platform from cp to the target (less 1 block), and stand on the
// target after its last action, as the search required. e: the bot's state now ({pos, vel, yaw,
// onGround, jumpTicks, collidedH}).
function verifyPlan (physics, world, e, plan, platforms, cp, target, extra) {
  let lo = Infinity
  for (let j = cp; j <= target; j++) lo = Math.min(lo, platforms[j][4] + 1)
  const c = {
    pos: new Vec3(e.pos[0], e.pos[1], e.pos[2]), vel: new Vec3(e.vel[0], e.vel[1], e.vel[2]), onGround: e.onGround,
    isInWater: false, isInLava: false, isInWeb: false, isCollidedHorizontally: !!e.collidedH, isCollidedVertically: false,
    elytraFlying: false, jumpTicks: e.jumpTicks, jumpQueued: false, fireworkRocketDuration: 0, attributes: {}, yaw: e.yaw,
    pitch: 0, control: {}, jumpBoost: 0, speed: 0, slowness: 0, dolphinsGrace: 0, slowFalling: 0, levitation: 0,
    depthStrider: 0, elytraEquipped: false
  }
  for (const a of plan) {
    c.yaw = wrapYaw(c.yaw + YAW_STEPS_DEG[a[4]] * Math.PI / 180)
    c.control = { forward: a[0] === 2, back: a[0] === 0, left: a[1] === 0, right: a[1] === 2, jump: a[2] === 1, sprint: a[3] === 1, sneak: false }
    physics.simulatePlayer(c, world)
    const s = { pos: [c.pos.x, c.pos.y, c.pos.z], onGround: c.onGround }
    if (onObstacle(s, extra) || s.pos[1] <= lo - 1.0) return false
  }
  return standsOn({ pos: [c.pos.x, c.pos.y, c.pos.z], onGround: c.onGround }, platforms[target])
}

module.exports = { Assist, DEFAULTS, PlannerService, serviceArgs, verifyPlan }
