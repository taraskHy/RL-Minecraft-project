'use strict'
// Greedy route planner: the bot finds its own route from the blocks it sees.
// A line-by-line port of parkour/planner.py (see its docstring); tests/test_planner.py
// checks that both pick the same platforms on the same courses.

const STAND_EPS = 1e-6
const HALF_WIDTH = 0.3
const PLATFORM_SLOTS = 4
const LOOKAHEAD = PLATFORM_SLOTS - 1
const MAX_GAP = { 1: 3, 0: 4, '-1': 4, '-2': 4, '-3': 5 }
const FORGIVENESS = {
  1: [0.593, 0.274, 0.124, 0.021],
  0: [0.62, 0.62, 0.196, 0.078, 0.01],
  '-1': [0.867, 0.876, 0.224, 0.104, 0.031],
  '-2': [0.859, 0.878, 0.369, 0.13, 0.044],
  '-3': [0.826, 0.859, 0.656, 0.156, 0.062, 0.021]
}
const DEFAULTS = {
  radius: 12, below: 10, above: 4, max_gap: 5.5, dy_min: -3, dy_max: 1, align: 2.0, slab_only: true,
  limits: 'table', slack: 0.5, cost: 'distance', mark_passed: true, clear_path: true, head_hitters: true
}

class Platform {
  constructor (cells) {
    this.cells = cells
    this.x0 = Infinity; this.x1 = -Infinity; this.z0 = Infinity; this.z1 = -Infinity
    for (const [x, , z] of cells) {
      this.x0 = Math.min(this.x0, x); this.x1 = Math.max(this.x1, x)
      this.z0 = Math.min(this.z0, z); this.z1 = Math.max(this.z1, z)
    }
    this.y = cells[0][1]
  }

  rect () { return [this.x0, this.x1, this.z0, this.z1, this.y] }
  centre () { return [(this.x0 + this.x1 + 1) / 2, this.y + 1, (this.z0 + this.z1 + 1) / 2] }
  contains ([x, y, z]) { return y === this.y && this.x0 <= x && x <= this.x1 && this.z0 <= z && z <= this.z1 }
}

const key = (x, y, z) => `${x},${y},${z}`
const norm = (a, b) => Math.sqrt(a * a + b * b)

function perceive (isSolid, cx, fy, cz, cfg) {
  const standable = (x, y, z) => {
    if (!isSolid(x, y, z) || isSolid(x, y + 1, z) || isSolid(x, y + 2, z)) return false
    return !(cfg.slab_only && isSolid(x, y - 1, z))
  }
  const r = cfg.radius
  const seen = new Set()
  const out = []
  for (let y = fy - cfg.below; y <= fy + cfg.above; y++) {
    for (let x = cx - r; x <= cx + r; x++) {
      for (let z = cz - r; z <= cz + r; z++) {
        if (seen.has(key(x, y, z)) || !standable(x, y, z)) continue
        const cells = [[x, y, z]]
        seen.add(key(x, y, z))
        let k = 0
        while (k < cells.length) {
          const [a, , b] = cells[k]
          k++
          for (const [nx, nz] of [[a - 1, b], [a + 1, b], [a, b - 1], [a, b + 1]]) {
            if (Math.abs(nx - cx) > r || Math.abs(nz - cz) > r || seen.has(key(nx, y, nz))) continue
            if (standable(nx, y, nz)) {
              seen.add(key(nx, y, nz))
              cells.push([nx, y, nz])
            }
          }
        }
        out.push(new Platform(cells))
      }
    }
  }
  return out
}

function pathClear (isSolid, a, b) {
  const ac = a.centre(); const bc = b.centre()
  const y0 = Math.min(a.y, b.y) + 1; const y1 = Math.max(a.y, b.y) + 2
  const n = Math.max(1, Math.ceil(norm(bc[0] - ac[0], bc[2] - ac[2]) / 0.25))
  for (let k = 1; k < n; k++) {
    const t = k / n
    const x = Math.floor(ac[0] + t * (bc[0] - ac[0]))
    const z = Math.floor(ac[2] + t * (bc[2] - ac[2]))
    if ((a.x0 <= x && x <= a.x1 && a.z0 <= z && z <= a.z1) || (b.x0 <= x && x <= b.x1 && b.z0 <= z && z <= b.z1)) continue
    for (let y = y0; y <= y1; y++) if (isSolid(x, y, z)) return false
  }
  return true
}

function goalCell (goal) { return [Math.floor(goal[0]), Math.round(goal[1]) - 1, Math.floor(goal[2])] }

function headHitter (p, platforms) {
  for (const q of platforms) {
    const dy = p.y - q.y
    if (dy === 4 && Math.max(0, p.x0 - (q.x1 + 1), q.x0 - (p.x1 + 1)) === 0 &&
        Math.max(0, p.z0 - (q.z1 + 1), q.z0 - (p.z1 + 1)) === 0) return true
  }
  return false
}

function choose (platforms, current, visited, goal, cfg, isSolid) {
  const chain = []
  let frm = current
  for (let step = 0; step < LOOKAHEAD; step++) {
    const fc = frm.centre()
    const gx = goal[0] - fc[0]; const gz = goal[2] - fc[2]
    const gn = norm(gx, gz)
    let best = null; let bestCost = Infinity
    for (const p of platforms) {
      if (p === current || chain.includes(p) || p.cells.some(c => visited.has(key(...c)))) continue
      const dy = p.y - frm.y
      if (dy < cfg.dy_min || dy > cfg.dy_max) continue
      const dx = Math.max(0, p.x0 - (frm.x1 + 1), frm.x0 - (p.x1 + 1))
      const dz = Math.max(0, p.z0 - (frm.z1 + 1), frm.z0 - (p.z1 + 1))
      const d = norm(dx, dz)
      if (d > (cfg.limits === 'table' ? MAX_GAP[dy] + cfg.slack : cfg.max_gap)) continue
      if (cfg.clear_path && isSolid && !pathClear(isSolid, frm, p)) continue
      const pc = p.centre()
      const vx = pc[0] - fc[0]; const vz = pc[2] - fc[2]
      const vn = norm(vx, vz)
      const cos = vn > 1e-9 && gn > 1e-9 ? (vx * gx + vz * gz) / (vn * gn) : 0.0
      let cost
      if (cfg.cost === 'difficulty') {
        const f = FORGIVENESS[dy]
        cost = -Math.log(f[Math.min(Math.round(d), f.length - 1)]) - cfg.align * cos
      } else {
        cost = d - cfg.align * cos
      }
      if (cost < bestCost - 1e-9) { best = p; bestCost = cost }
    }
    if (best === null) break
    chain.push(best)
    if (best.contains(goalCell(goal))) break
    frm = best
  }
  return chain
}

function supportCells ([x, y, z]) {
  const cy = Math.floor(y + STAND_EPS) - 1
  const out = []
  for (let ix = Math.floor(x - HALF_WIDTH); ix < Math.ceil(x + HALF_WIDTH); ix++) {
    for (let iz = Math.floor(z - HALF_WIDTH); iz < Math.ceil(z + HALF_WIDTH); iz++) out.push([ix, cy, iz])
  }
  return out
}

class GreedyPlanner {
  constructor (isSolid, goal, cfg = {}) {
    this.isSolid = isSolid
    this.goal = goal
    this.cfg = { ...DEFAULTS, ...cfg }
    this.visited = new Set()
    this.current = null
    this.chain = []
    this.startDist = null
    this.progress = 0.0
    this.events = []
  }

  start (pos) {
    let cell = supportCells(pos).find(c => this.isSolid(...c))
    if (cell === undefined) cell = [Math.floor(pos[0]), Math.floor(pos[1] + STAND_EPS) - 1, Math.floor(pos[2])]
    this._replan(cell, pos, 0)
  }

  update (pos, onGround, tick) {
    if (!onGround || Math.abs(pos[1] - Math.round(pos[1])) > STAND_EPS) return false
    const cells = supportCells(pos).filter(c => this.isSolid(...c))
    if (!cells.length || (this.current !== null && cells.some(c => this.current.contains(c)))) return false
    return this._replan(cells[0], pos, tick)
  }

  _replan (cell, pos, tick) {
    const cfg = this.cfg
    const platforms = perceive(this.isSolid, Math.floor(pos[0]), cell[1] + 1, Math.floor(pos[2]), cfg)
    const here = platforms.find(p => p.contains(cell))
    if (here === undefined) return false
    if (cfg.mark_passed) {
      const j = this.chain.findIndex(p => p.contains(cell))
      if (j >= 0) for (const q of this.chain.slice(0, j)) for (const c of q.cells) this.visited.add(key(...c))
    }
    this.current = here
    for (const c of here.cells) this.visited.add(key(...c))
    const targets = cfg.head_hitters ? platforms.filter(p => !headHitter(p, platforms)) : platforms
    this.chain = choose(targets, here, this.visited, this.goal, cfg, this.isSolid)
    const c = here.centre()
    const dx = c[0] - this.goal[0]; const dy = c[1] - this.goal[1]; const dz = c[2] - this.goal[2]
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (this.startDist === null) this.startDist = Math.max(dist, 1e-9)
    this.progress = Math.min(1.0, Math.max(0.0, 1.0 - dist / this.startDist))
    this.events.push([tick, here.rect(), this.chain.map(p => p.rect())])
    return true
  }

  // Slots for the policy, current platform first: [{ p: rect, goal: bool }].
  slots () {
    const ps = (this.current !== null ? [this.current] : []).concat(this.chain)
    const gcell = goalCell(this.goal)
    return ps.map(p => ({ p: p.rect(), goal: p.contains(gcell) }))
  }
}

module.exports = { GreedyPlanner, perceive, choose, DEFAULTS, headHitter, goalCell, supportCells, MAX_GAP }
