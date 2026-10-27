'use strict'
// Vanilla Minecraft's 1e-7 collision tolerance for prismarine-physics.
//
// prismarine-physics 1.11.1 compares block faces exactly. A player resting
// against a face can then read as overlapping it after rounding
// (32.3 - 0.3 = 31.999999999999996), the engine stops blocking that block,
// and the player slips into its corner. Vanilla's VoxelShape.collide treats
// faces closer than 1e-7 as touching. Requiring this module patches the
// shared AABB class, so Mineflayer's own physics (same module instance) gets
// the same rule as the simulator (parkour/physics.py, COLLISION_EPSILON).

const path = require('path')
const AABB = require(path.join(path.dirname(require.resolve('prismarine-physics')), 'lib', 'aabb'))

const EPS = 1e-7

function limit (offset, gap, positive) {
  // gap: distance to the block face along the move (>= 0 when moving +, <= 0 when moving -)
  if (positive) {
    let c = Math.max(gap, 0)
    if (c < EPS) c = 0
    return c < offset ? c : offset
  }
  let c = Math.min(gap, 0)
  if (c > -EPS) c = 0
  return c > offset ? c : offset
}

AABB.prototype.computeOffsetX = function (other, offsetX) {
  if (other.maxY > this.minY + EPS && other.minY < this.maxY - EPS && other.maxZ > this.minZ + EPS && other.minZ < this.maxZ - EPS) {
    if (offsetX > 0.0 && other.maxX <= this.minX + EPS) return limit(offsetX, this.minX - other.maxX, true)
    if (offsetX < 0.0 && other.minX >= this.maxX - EPS) return limit(offsetX, this.maxX - other.minX, false)
  }
  return offsetX
}

AABB.prototype.computeOffsetY = function (other, offsetY) {
  if (other.maxX > this.minX + EPS && other.minX < this.maxX - EPS && other.maxZ > this.minZ + EPS && other.minZ < this.maxZ - EPS) {
    if (offsetY > 0.0 && other.maxY <= this.minY + EPS) return limit(offsetY, this.minY - other.maxY, true)
    if (offsetY < 0.0 && other.minY >= this.maxY - EPS) return limit(offsetY, this.maxY - other.minY, false)
  }
  return offsetY
}

AABB.prototype.computeOffsetZ = function (other, offsetZ) {
  if (other.maxX > this.minX + EPS && other.minX < this.maxX - EPS && other.maxY > this.minY + EPS && other.minY < this.maxY - EPS) {
    if (offsetZ > 0.0 && other.maxZ <= this.minZ + EPS) return limit(offsetZ, this.minZ - other.maxZ, true)
    if (offsetZ < 0.0 && other.minZ >= this.maxZ - EPS) return limit(offsetZ, this.maxZ - other.minZ, false)
  }
  return offsetZ
}

module.exports = { EPS, AABB }
