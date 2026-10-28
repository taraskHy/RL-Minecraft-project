'use strict'
// Collision boxes of the block states the simulator can use besides full cubes (physics.World(shape=...)),
// read from prismarine-block for Minecraft 1.21.4, the data Mineflayer's physics collides with.
//
//   node export_shapes.js [--check]      writes ../configs/block_shapes.json (--check: compare only)
//
// Isolated states only: fences, walls and panes change shape with their neighbours, so courses use
// them as single posts. Ladders are here for their boxes; their climbing rule is World(climb=...).
// Other blocks with movement rules of their own (slime, honey, water, webs, scaffolding) are not.

const fs = require('fs')
const path = require('path')
const mcData = require('minecraft-data')('1.21.4')
// prismarine-block is a dependency of Mineflayer, not of the bridge itself.
const Block = require(require.resolve('prismarine-block', { paths: [path.dirname(require.resolve('mineflayer'))] }))('1.21.4')

const KINDS = [
  ['full', 'stone', {}],
  ['bottom_slab', 'stone_slab', { type: 'bottom', waterlogged: false }],
  ['top_slab', 'stone_slab', { type: 'top', waterlogged: false }],
  ['soul_sand', 'soul_sand', {}],
  ['dirt_path', 'dirt_path', {}],
  ['enchanting_table', 'enchanting_table', {}],
  ['daylight_detector', 'daylight_detector', { inverted: false, power: 0 }],
  ['bottom_trapdoor', 'oak_trapdoor', { facing: 'north', half: 'bottom', open: false, powered: false, waterlogged: false }],
  ['top_trapdoor', 'oak_trapdoor', { facing: 'north', half: 'top', open: false, powered: false, waterlogged: false }],
  ['carpet', 'white_carpet', {}],
  ['fence_post', 'oak_fence', { north: false, east: false, south: false, west: false, waterlogged: false }],
  ['wall_post', 'cobblestone_wall', { up: true, north: 'none', east: 'none', south: 'none', west: 'none', waterlogged: false }],
  ['iron_bars', 'iron_bars', { north: false, east: false, south: false, west: false, waterlogged: false }],
  ['end_rod', 'end_rod', { facing: 'up' }],
  ['stairs_north', 'stone_stairs', { facing: 'north', half: 'bottom', shape: 'straight', waterlogged: false }],
  ['stairs_east', 'stone_stairs', { facing: 'east', half: 'bottom', shape: 'straight', waterlogged: false }],
  ['stairs_south', 'stone_stairs', { facing: 'south', half: 'bottom', shape: 'straight', waterlogged: false }],
  ['stairs_west', 'stone_stairs', { facing: 'west', half: 'bottom', shape: 'straight', waterlogged: false }],
  // Ladders (climbable: physics.World(climb=...)); the facing is the side the ladder is seen from.
  ['ladder_north', 'ladder', { facing: 'north', waterlogged: false }],
  ['ladder_east', 'ladder', { facing: 'east', waterlogged: false }],
  ['ladder_south', 'ladder', { facing: 'south', waterlogged: false }],
  ['ladder_west', 'ladder', { facing: 'west', waterlogged: false }]
]

function table () {
  return KINDS.map(([name, block, properties]) => {
    const b = Block.fromProperties(mcData.blocksByName[block].id, properties, 0)
    return { name, block, properties, state: b.stateId, boxes: b.shapes }
  })
}

module.exports = { KINDS, table }

if (require.main === module) {
  const file = path.join(__dirname, '..', 'configs', 'block_shapes.json')
  const data = { source: 'bridge/export_shapes.js: prismarine-block, Minecraft 1.21.4 (minecraft-data ' + require('minecraft-data/package.json').version + ')', shapes: table() }
  const text = JSON.stringify(data, null, 1) + '\n'
  if (process.argv.includes('--check')) {
    const same = fs.readFileSync(file, 'utf8') === text
    process.stdout.write(JSON.stringify({ same }) + '\n')
    process.exit(same ? 0 : 1)
  }
  fs.writeFileSync(file, text)
  process.stdout.write(`${data.shapes.length} shapes -> ${file}\n`)
}
