'use strict'
// Recomputes observations, standing/fall rules and policy logits for states
// recorded from the Python environment, so tests can compare the two.
//
// Input (file path argument, JSON):
//   { "policy": "path/to/policy.json" | null,
//     "courses": [ { "platforms": [[x0,x1,z0,z1,y], ...] } ],
//     "cases": [ { "course": 0, "state": {pos, vel, yaw, onGround, jumpTicks, collidedH},
//                  "lastAction": [..5], "cp": 0 } ] }
// Output (stdout, JSON): { "obs": [[...]], "logits": [[...]] | null, "standing": [..], "fell": [..] }

const fs = require('fs')
const { observe, standingOn, fellBelow, Policy } = require('./policy')

const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
const solids = input.courses.map(c => {
  const set = new Set()
  for (const [x0, x1, z0, z1, y] of c.platforms) {
    for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) set.add(`${x},${y},${z}`)
  }
  return set
})
const policy = input.policy ? Policy.load(input.policy) : null
const obs = []
const logits = []
const standing = []
const fell = []
for (const c of input.cases) {
  const platforms = input.courses[c.course].platforms
  const set = solids[c.course]
  const o = observe(c.state, c.lastAction, platforms, c.cp, (x, y, z) => set.has(`${x},${y},${z}`))
  obs.push(Array.from(o))
  if (policy) logits.push(Array.from(policy.logits(o)))
  standing.push(standingOn(c.state, platforms))
  fell.push(fellBelow(c.state, platforms, c.cp))
}
process.stdout.write(JSON.stringify({ obs, logits: policy ? logits : null, standing, fell }))
