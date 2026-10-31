'use strict'
// Learned route policy for the bot: a port of parkour/route_env.py (Scene.tokens, RouteRunner.decide)
// and of the route policy's actor (parkour/route_policy.py, exported by RoutePolicy.export).
// Perception, jump limits and path checks are bridge/planner.js's own; tests/test_route.py checks
// that the offline bot and the Python evaluation make the same plans with the same outcomes.

const fs = require('fs')
const { DEFAULTS, perceive, headHitter, goalCell, supportCells, MAX_GAP } = require('./planner')

const STAND_EPS = 1e-6
const LOOKAHEAD = 3
const norm = (a, b) => Math.sqrt(a * a + b * b)
const key = (x, y, z) => `${x},${y},${z}`
const centre = (p) => [(p.x0 + p.x1 + 1) / 2, p.y + 1, (p.z0 + p.z1 + 1) / 2]
const airGap = (a, b) => norm(Math.max(0, b.x0 - (a.x1 + 1), a.x0 - (b.x1 + 1)), Math.max(0, b.z0 - (a.z1 + 1), a.z0 - (b.z1 + 1)))
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// planner.path_clear, and whether a block sits 1 or 2 rows above its body-height band (route_env.path_clear);
// with the platforms in view T, also whether a block in the way is no part of one of them (route_env.wall_blocked).
function pathInfo (isSolid, a, b, T = null) {
  const ac = centre(a); const bc = centre(b)
  const y0 = Math.min(a.y, b.y) + 1; const y1 = Math.max(a.y, b.y) + 2
  const n = Math.max(1, Math.ceil(norm(bc[0] - ac[0], bc[2] - ac[2]) / 0.25))
  let clear = true; let head = false; let wall = false
  for (let k = 1; k < n; k++) {
    const t = k / n
    const x = Math.floor(ac[0] + t * (bc[0] - ac[0]))
    const z = Math.floor(ac[2] + t * (bc[2] - ac[2]))
    if ((a.x0 <= x && x <= a.x1 && a.z0 <= z && z <= a.z1) || (b.x0 <= x && x <= b.x1 && b.z0 <= z && z <= b.z1)) continue
    for (let y = y0; y <= y1; y++) {
      if (!isSolid(x, y, z)) continue
      clear = false
      if (T && !T.some(p => p.y === y && p.x0 <= x && x <= p.x1 && p.z0 <= z && z <= p.z1)) wall = true
    }
    if (isSolid(x, y1 + 1, z) || isSolid(x, y1 + 2, z)) head = true
  }
  return [clear, head, wall]
}

// Exact GELU needs erf: series below 3, continued fraction for erfc above (both to ~1e-15).
function erf (x) {
  const ax = Math.abs(x)
  if (ax < 3) {
    let term = ax; let sum = ax; const x2 = ax * ax
    for (let n = 1; n < 200; n++) {
      term *= -x2 / n
      const add = term / (2 * n + 1)
      sum += add
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break
    }
    const r = 2 / Math.sqrt(Math.PI) * sum
    return x < 0 ? -r : r
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...)))) (Lentz)
  let f = ax; let C = ax; let D = 0
  for (let n = 1; n < 300; n++) {
    const an = n / 2
    D = ax + an * D; D = 1 / D
    C = ax + an / C
    const delta = C * D
    f *= delta
    if (Math.abs(delta - 1) < 1e-16) break
  }
  const r = 1 - Math.exp(-ax * ax) / Math.sqrt(Math.PI) / f
  return x < 0 ? -r : r
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2))
const elu = (x) => (x > 0 ? x : Math.expm1(x))

function linear (x, { w, b }) {
  const y = new Float64Array(b.length)
  for (let r = 0; r < b.length; r++) {
    const row = w[r]
    let acc = b[r]
    for (let c = 0; c < x.length; c++) acc += row[c] * x[c]
    y[r] = acc
  }
  return y
}

function layerNorm (x, { w, b }, eps) {
  let mean = 0
  for (const v of x) mean += v
  mean /= x.length
  let v2 = 0
  for (const v of x) v2 += (v - mean) * (v - mean)
  const inv = 1 / Math.sqrt(v2 / x.length + eps)
  const y = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i]
  return y
}

class RouterNet {
  constructor (data) {
    if (data.format !== 'parkour-router-v1') throw new Error('unknown router format ' + data.format)
    Object.assign(this, data)
  }

  static load (path) { return new RouterNet(JSON.parse(fs.readFileSync(path, 'utf8'))) }

  // Logits of the tokens (list of feature arrays); not-selectable tokens get -1e9.
  logits (tokens, ok) {
    const d = this.d; const H = this.heads; const hd = d / H; const eps = this.ln_eps
    let xs = tokens.map(t => {
      const h = linear(Float64Array.from(t), this.embed[0])
      for (let i = 0; i < h.length; i++) h[i] = elu(h[i])
      return linear(h, this.embed[1])
    })
    const n = xs.length
    for (const L of this.layers) {
      const xn = xs.map(x => layerNorm(x, L.ln1, eps))
      const qkv = xn.map(x => linear(x, { w: L.in_w, b: L.in_b }))
      const att = []
      for (let i = 0; i < n; i++) {
        const o = new Float64Array(d)
        for (let h = 0; h < H; h++) {
          const s = new Float64Array(n)
          let m = -Infinity
          for (let j = 0; j < n; j++) {
            let acc = 0
            for (let c = 0; c < hd; c++) acc += qkv[i][h * hd + c] * qkv[j][d + h * hd + c]
            s[j] = acc / Math.sqrt(hd)
            if (s[j] > m) m = s[j]
          }
          let z = 0
          for (let j = 0; j < n; j++) { s[j] = Math.exp(s[j] - m); z += s[j] }
          for (let j = 0; j < n; j++) {
            const p = s[j] / z
            for (let c = 0; c < hd; c++) o[h * hd + c] += p * qkv[j][2 * d + h * hd + c]
          }
        }
        att.push(linear(o, L.out))
      }
      xs = xs.map((x, i) => x.map((v, c) => v + att[i][c]))
      xs = xs.map(x => {
        const h = linear(layerNorm(x, L.ln2, eps), L.ff1)
        for (let i = 0; i < h.length; i++) h[i] = gelu(h[i])
        const f = linear(h, L.ff2)
        return x.map((v, c) => v + f[c])
      })
    }
    return xs.map((x, i) => (ok[i] ? linear(layerNorm(x, this.out_norm, eps), this.pi)[0] : -1e9))
  }

  argmax (tokens, ok) {
    const lg = this.logits(tokens, ok)
    let best = -1; let bv = -Infinity
    for (let i = 0; i < lg.length; i++) if (ok[i] && lg[i] > bv) { bv = lg[i]; best = i }
    return best
  }
}

// The planner's options from token ``frm`` (route_env.Scene.options): selectable flags and the
// greedy cost, air gap, height change, cosine and head flag of every token. With cfg.blocked "keep" or
// "plan" (RouteConfig.blocked) a token whose straight path is blocked stays selectable, flagged ``blocked``,
// and ``walled`` if a block in the way is no part of a platform in view (Scene.walled: a neo).
function options (T, frm, excluded, goal, cfg, isSolid) {
  const keep = cfg.blocked === 'keep' || cfg.blocked === 'plan'
  const f = T[frm]; const fc = centre(f)
  const gx = goal[0] - fc[0]; const gz = goal[2] - fc[2]; const gn = norm(gx, gz)
  return T.map((p, i) => {
    const dy = p.y - f.y
    const d = airGap(f, p)
    const pc = centre(p)
    const vx = pc[0] - fc[0]; const vz = pc[2] - fc[2]; const vn = norm(vx, vz)
    const cos = vn > 1e-9 && gn > 1e-9 ? (vx * gx + vz * gz) / (vn * gn) : 0.0
    const cost = d - cfg.align * cos
    let ok = i !== frm && !excluded[i] && dy >= cfg.dy_min && dy <= cfg.dy_max &&
      d <= (cfg.limits === 'table' ? MAX_GAP[dy] + cfg.slack : cfg.max_gap)
    let head = false
    let blocked = false
    let walled = false
    if (ok && cfg.clear_path) {
      const [clear, h, w] = pathInfo(isSolid, f, p, keep ? T : null)
      if (keep) {
        blocked = !clear
        walled = w
        head = h
      } else {
        ok = clear
        head = h && clear
      }
    }
    return { ok, cost, d, dy, cos, head, blocked, walled }
  })
}

function greedyPick (opts) {
  let best = Infinity
  for (const o of opts) if (o.ok && o.cost < best) best = o.cost
  for (let i = 0; i < opts.length; i++) if (opts[i].ok && opts[i].cost <= best + 1e-9) return i
  return -1
}

// Token features (route_env.Scene.tokens), same order and scaling.
function tokens (T, frm, here, excluded, visited, chain, depth, progress, goal, isGoal, edge, cfg, isSolid, player, timing) {
  const opts = options(T, frm, excluded, goal, cfg, isSolid)
  const f = T[frm]; const fc = centre(f)
  const gx = goal[0] - fc[0]; const gz = goal[2] - fc[2]; const gn = Math.sqrt(gx * gx + gz * gz)
  const ux = gn > 1e-9 ? gx / Math.max(gn, 1e-9) : 1; const uz = gn > 1e-9 ? gz / Math.max(gn, 1e-9) : 0
  const vx = -uz; const vz = ux
  const frame = (x, z) => { const dx = x - fc[0]; const dz = z - fc[2]; return [dx * ux + dz * uz, dx * vx + dz * vz] }
  const n = T.length
  const targetOk = T.map((p, j) => !excluded[j] && j !== frm)
  const nOn = T.map((p, i) => {
    let c = 0
    for (let j = 0; j < n; j++) {
      if (j === i || !targetOk[j]) continue
      const dy = T[j].y - p.y
      if (dy < cfg.dy_min || dy > cfg.dy_max) continue
      if (airGap(p, T[j]) <= MAX_GAP[Math.max(-3, Math.min(1, dy))] + cfg.slack) c++
    }
    return c
  })
  // Stable rank by cost among the selectable tokens.
  const order = [...Array(n).keys()].sort((a, b) => {
    const ca = opts[a].ok ? opts[a].cost : Infinity; const cb = opts[b].ok ? opts[b].cost : Infinity
    return ca < cb ? -1 : ca > cb ? 1 : a - b
  })
  const rank = new Array(n)
  order.forEach((t, r) => { rank[t] = r })
  const g = greedyPick(opts)
  const nSel = opts.filter(o => o.ok).length
  const nVis = visited.filter(v => v).length
  let pu = 0; let pv = 0; let vu = 0; let vv = 0
  if (player && depth === 0) {
    const dx = player.pos[0] - fc[0]; const dz = player.pos[2] - fc[2]
    pu = (dx * ux + dz * uz) / 2; pv = (dx * vx + dz * vz) / 2
    vu = (player.vel[0] * ux + player.vel[2] * uz) / 0.4; vv = (player.vel[0] * vx + player.vel[2] * vz) / 0.4
  }
  const tNew = timing ? Math.min(timing[0] / 200, 3) : 0
  const tDec = timing ? Math.min(timing[1] / 50, 3) : 0
  const glob = [gn / 20, (goal[1] - fc[1]) / 10, depth === 0 ? 1 : 0, depth === 1 ? 1 : 0, depth === 2 ? 1 : 0,
    nSel / 8, nVis / 10, progress, pu, pv, vu, vv, tNew, tDec]
  const feats = T.map((p, i) => {
    const o = opts[i]
    const pc = centre(p)
    const [cu, cv] = frame(pc[0], pc[2])
    const xs = [p.x0, p.x1 + 1, p.x1 + 1, p.x0]; const zs = [p.z0, p.z0, p.z1 + 1, p.z1 + 1]
    const k = xs.map((x, c) => frame(x, zs[c]))
    const hx = goal[0] - pc[0]; const hz = goal[2] - pc[2]
    const dyf = p.y - f.y
    return [cu / 8, cv / 8, dyf / 3, Math.min(o.d, 12) / 4, (p.x1 - p.x0 + 1) / 4, (p.z1 - p.z0 + 1) / 4, o.cos,
      clamp(o.cost, -3, 12) / 4, (o.ok ? rank[i] : 16) / 16, i === g ? 1 : 0, Math.sqrt(hx * hx + hz * hz) / 20,
      (goal[1] - pc[1]) / 10, isGoal[i] ? 1 : 0, edge[i] ? 1 : 0, nOn[i] / 4, nOn[i] === 0 ? 1 : 0,
      o.ok ? 1 : 0, i === frm ? 1 : 0, i === here ? 1 : 0, visited[i] ? 1 : 0, chain[i] ? 1 : 0,
      dyf >= cfg.dy_min && dyf <= cfg.dy_max ? 1 : 0, o.head ? 1 : 0,
      ...k.map(q => q[0] / 8), ...k.map(q => q[1] / 8), ...glob]
  })
  return { feats, ok: opts.map(o => o.ok), opts }
}

// Learned router with GreedyPlanner's interface (start, update, slots, progress, events).
class LearnedRouter {
  constructor (isSolid, goal, net, cfg = {}, fill = 'greedy') {
    this.isSolid = isSolid
    this.goal = goal
    this.net = net
    this.cfg = { ...DEFAULTS, ...cfg }
    this.fill = fill
    this.visited = new Set()
    this.current = null
    this.chain = []
    this.startDist = null
    this.progress = 0.0
    this.events = []
    this.newTick = 0
    this.decTick = 0
    this.decisions = 0
    this.walledChoice = false // the last decision's first choice is blocked by a wall (cfg.blocked "plan": a neo)
    this.walledAhead = false // a later slot of its route is (RouteRunner.decide's walled_chain; plan_ahead)
  }

  start (pos, s) {
    let cell = supportCells(pos).find(c => this.isSolid(...c))
    if (cell === undefined) cell = [Math.floor(pos[0]), Math.floor(pos[1] + STAND_EPS) - 1, Math.floor(pos[2])]
    this._replan(cell, pos, 0, s)
  }

  update (pos, onGround, tick, s) {
    if (!onGround || Math.abs(pos[1] - Math.round(pos[1])) > STAND_EPS) return false
    const cells = supportCells(pos).filter(c => this.isSolid(...c))
    if (!cells.length || (this.current !== null && cells.some(c => this.current.contains(c)))) return false
    return this._replan(cells[0], pos, tick, s)
  }

  _replan (cell, pos, tick, s) {
    const cfg = this.cfg
    const cx = Math.floor(pos[0]); const cz = Math.floor(pos[2]); const r = cfg.radius
    const platforms = perceive(this.isSolid, cx, cell[1] + 1, cz, cfg)
    const T = (cfg.head_hitters ? platforms.filter(p => !headHitter(p, platforms)) : platforms)
      .sort((a, b) => a.y - b.y || a.x0 - b.x0 || a.z0 - b.z0)
    const here = T.findIndex(p => p.contains(cell))
    if (here < 0) return false
    if (cfg.mark_passed) {
      const j = this.chain.findIndex(p => p.contains(cell))
      if (j >= 0) for (const q of this.chain.slice(0, j)) for (const c of q.cells) this.visited.add(key(...c))
    }
    const wasVisited = T[here].cells.some(c => this.visited.has(key(...c)))
    const timing = [tick - this.newTick, tick - this.decTick]
    if (!wasVisited) this.newTick = tick
    this.decTick = tick
    this.current = T[here]
    for (const c of T[here].cells) this.visited.add(key(...c))
    const hc = centre(T[here])
    const dx = hc[0] - this.goal[0]; const dy = hc[1] - this.goal[1]; const dz = hc[2] - this.goal[2]
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (this.decisions === 0) this.startDist = Math.max(dist, 1e-9)
    this.progress = Math.min(1.0, Math.max(0.0, 1.0 - dist / this.startDist))
    this.decisions++
    const gcell = goalCell(this.goal)
    const isGoal = T.map(p => p.contains(gcell))
    const edge = T.map(p => p.x0 === cx - r || p.x1 === cx + r || p.z0 === cz - r || p.z1 === cz + r)
    const visited = T.map(p => p.cells.some(c => this.visited.has(key(...c))))
    const excluded = visited.slice()
    excluded[here] = true
    const chain = T.map(() => false)
    const player = s ? { pos: s.pos, vel: s.vel } : { pos, vel: [0, 0, 0] }
    const { feats, ok, opts } = tokens(T, here, here, excluded, visited, chain, 0, this.progress, this.goal, isGoal, edge, cfg, this.isSolid, player, timing)
    const picks = []
    let prev = ok.some(v => v) ? this.net.argmax(feats, ok) : -1
    if (prev >= 0) picks.push(prev)
    this.walledChoice = prev >= 0 && opts[prev].walled
    this.walledAhead = false
    for (let depth = 1; depth <= 2 && prev >= 0 && !isGoal[prev]; depth++) {
      chain[prev] = true
      const ex = excluded.map((v, i) => v || chain[i])
      let nxt, o
      if (this.fill === 'policy') {
        const t = tokens(T, prev, here, ex, visited, chain, depth, this.progress, this.goal, isGoal, edge, cfg, this.isSolid, null, timing)
        nxt = t.ok.some(v => v) ? this.net.argmax(t.feats, t.ok) : -1
        o = t.opts
      } else {
        o = options(T, prev, ex, this.goal, cfg, this.isSolid)
        nxt = greedyPick(o)
      }
      if (nxt < 0) break
      if (o[nxt].walled) this.walledAhead = true
      picks.push(nxt)
      prev = nxt
    }
    this.chain = picks.map(i => T[i])
    this.events.push([tick, T[here].rect(), this.chain.map(p => p.rect())])
    return true
  }

  slots () {
    const ps = (this.current !== null ? [this.current] : []).concat(this.chain)
    const gcell = goalCell(this.goal)
    return ps.map(p => ({ p: p.rect(), goal: p.contains(gcell) }))
  }
}

module.exports = { LearnedRouter, RouterNet, tokens, erf }
