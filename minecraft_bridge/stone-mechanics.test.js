'use strict'

// The facts of section 1 of docs/stone_tools_design.md, checked against the
// pinned minecraft-data (pc 1.21.4) and libraries, resolved through Mineflayer.
const test = require('node:test')
const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const { Vec3 } = require('vec3')
const { STONE_SOURCES, STONE_DROPS, STONE_MATERIALS, STONE_LOOKALIKES, STONE_PICKAXE_REQUIREMENT, WOODEN_PICKAXE, isStoneMaterial } = require('./stone-helpers')

const mineflayerEntry = require.resolve('mineflayer')
const fromMineflayer = name => require(require.resolve(name, { paths: [mineflayerEntry] }))
const registry = fromMineflayer('prismarine-registry')('1.21.4')
const Block = fromMineflayer('prismarine-block')(registry)
const Item = fromMineflayer('prismarine-item')(registry)
const Recipe = fromMineflayer('prismarine-recipe')(registry).Recipe
const item = name => registry.itemsByName[name]
const block = name => Block.fromStateId(registry.blocksByName[name].defaultState, 0)
const dropNames = data => (data.drops || []).map(drop => registry.items[typeof drop === 'object' ? (drop.drop?.id ?? drop.id) : drop]?.name)
const harvestNames = data => data.harvestTools ? Object.keys(data.harvestTools).map(id => registry.items[id].name).sort() : null

test('the pinned data are minecraft-data 3.117.0 for pc 1.21.4', () => {
  const version = require(require.resolve('minecraft-data/package.json', { paths: [mineflayerEntry] })).version
  assert.equal(version, '3.117.0')
  assert.equal(registry.version.minecraftVersion, '1.21.4')
  assert.equal(registry.version.type, 'pc')
})

test('there are exactly three stone pickaxe recipes, all 3x3, one material across the top and two sticks down the middle', () => {
  const pickaxe = item('stone_pickaxe')
  assert.equal(pickaxe.id, 856)
  const raw = registry.recipes[pickaxe.id]
  assert.equal(raw.length, 3)
  const stick = item('stick').id
  const materials = raw.map(recipe => {
    assert.equal(recipe.inShape.length, 3)
    assert(recipe.inShape.every(row => row.length === 3))
    assert.deepEqual(recipe.result, { id: pickaxe.id, count: 1 })
    assert.deepEqual(recipe.inShape.slice(1), [[null, stick, null], [null, stick, null]])
    const top = recipe.inShape[0]
    assert(top.every(id => id === top[0]))
    return registry.items[top[0]].name
  })
  assert.deepEqual(materials.sort(), ['blackstone', 'cobbled_deepslate', 'cobblestone'])
  assert.deepEqual([...STONE_MATERIALS].sort(), materials)
  // prismarine-recipe, which bot.recipesFor uses, needs a crafting table for all three.
  const recipes = Recipe.find(pickaxe.id)
  assert.equal(recipes.length, 3)
  assert(recipes.every(recipe => recipe.requiresTable))
  // No recipe mixes materials, so a mixed grid needs the fill.
  assert(recipes.every(recipe => new Set(recipe.inShape.flat().filter(slot => slot.id !== -1 && slot.id !== stick).map(slot => slot.id)).size === 1))
  // The coded requirement is what every recipe consumes.
  for (const recipe of recipes) {
    const consumed = recipe.delta.filter(entry => entry.count < 0)
    assert.deepEqual(consumed.map(entry => [isStoneMaterial(registry.items[entry.id].name) ? 'material' : registry.items[entry.id].name, -entry.count]).sort(),
      [['material', STONE_PICKAXE_REQUIREMENT.materials], ['stick', STONE_PICKAXE_REQUIREMENT.sticks]])
  }
  assert.equal(STONE_PICKAXE_REQUIREMENT.table, true)
})

test('exactly five blocks give a recipe material to a wooden pickaxe, with the hardness of section 1', () => {
  const materials = new Set(STONE_MATERIALS)
  const givers = registry.blocksArray.filter(data => dropNames(data).some(name => materials.has(name)) && (!data.harvestTools || data.harvestTools[WOODEN_PICKAXE]))
  const facts = Object.fromEntries(givers.map(data => [data.name, { hardness: data.hardness, drops: dropNames(data), material: data.material }]))
  assert.deepEqual(facts, {
    stone: { hardness: 1.5, drops: ['cobblestone'], material: 'mineable/pickaxe' },
    cobblestone: { hardness: 2, drops: ['cobblestone'], material: 'mineable/pickaxe' },
    deepslate: { hardness: 3, drops: ['cobbled_deepslate'], material: 'mineable/pickaxe' },
    cobbled_deepslate: { hardness: 3.5, drops: ['cobbled_deepslate'], material: 'mineable/pickaxe' },
    blackstone: { hardness: 1.5, drops: ['blackstone'], material: 'mineable/pickaxe' }
  })
  for (const [source, drop] of Object.entries(STONE_DROPS)) assert.deepEqual(dropNames(registry.blocksByName[source]), [drop], source)
  for (const source of STONE_SOURCES) assert(harvestNames(registry.blocksByName[source]).includes('wooden_pickaxe'), source)
  // By hand none of them drops anything: the harvest tools are pickaxes only.
  for (const name of Object.keys(facts)) assert.equal(Boolean(block(name).canHarvest(null)), false, name)
})

test('lookalikes drop themselves and infested stone drops nothing', () => {
  for (const name of STONE_LOOKALIKES) {
    const data = registry.blocksByName[name]
    assert.deepEqual(dropNames(data), [name], name)
    assert.equal(data.material, 'mineable/pickaxe', name)
    assert(harvestNames(data).includes('wooden_pickaxe'), name)
  }
  assert.deepEqual(STONE_LOOKALIKES.map(name => registry.blocksByName[name].hardness), [1.5, 1.5, 1.5, 1.5, 0.75, 0.8, 1.25, 2])
  const infested = registry.blocksByName.infested_stone
  assert.deepEqual([infested.hardness, dropNames(infested), infested.harvestTools ?? null, infested.material], [0.75, [], null, 'mineable/pickaxe'])
  assert.deepEqual(registry.blocksArray.filter(data => data.name.startsWith('infested_')).map(data => [data.name, data.hardness, dropNames(data).length]), [
    ['infested_stone', 0.75, 0], ['infested_cobblestone', 1, 0], ['infested_stone_bricks', 0.75, 0], ['infested_mossy_stone_bricks', 0.75, 0],
    ['infested_cracked_stone_bricks', 0.75, 0], ['infested_chiseled_stone_bricks', 0.75, 0], ['infested_deepslate', 1.5, 0]])
})

test('the wooden pickaxe lasts 59 uses at speed 2, the stone pickaxe 131, and iron and copper ore need stone', () => {
  assert.equal(item('wooden_pickaxe').maxDurability, 59)
  assert.equal(item('stone_pickaxe').maxDurability, 131)
  assert.equal(registry.materials['mineable/pickaxe'][WOODEN_PICKAXE], 2)
  assert.equal(registry.materials['mineable/pickaxe'][item('stone_pickaxe').id], 4)
  for (const name of ['iron_ore', 'copper_ore', 'deepslate_iron_ore', 'deepslate_copper_ore']) {
    const data = registry.blocksByName[name]
    assert.equal(data.material, 'incorrect_for_wooden_tool', name)
    const tools = harvestNames(data)
    assert(tools.includes('stone_pickaxe') && !tools.includes('wooden_pickaxe'), name)
  }
  assert.deepEqual(harvestNames(registry.blocksByName.obsidian), ['diamond_pickaxe', 'netherite_pickaxe'])
})

test('a furnace takes 8 of one stone-tool material', () => {
  const raw = registry.recipes[item('furnace').id]
  assert.equal(raw.length, 3)
  const materials = raw.map(recipe => {
    const ids = recipe.inShape.flat()
    assert.equal(ids.filter(id => id === null).length, 1)
    assert.equal(recipe.inShape[1][1], null)
    const used = ids.filter(id => id !== null)
    assert.equal(used.length, 8)
    assert.equal(new Set(used).size, 1)
    return registry.items[used[0]].name
  })
  assert.deepEqual(materials.sort(), ['blackstone', 'cobbled_deepslate', 'cobblestone'])
})

test('dig times from the vanilla formula, through block.digTime and through Mineflayer\'s bot.digTime', () => {
  const wooden = WOODEN_PICKAXE
  const time = (name, held, onGround = true) => block(name).digTime(held, false, false, !onGround, [], {})
  // Measured values (ms) beside the design's estimates in the comments.
  assert.equal(time('stone', wooden), 1150) // about 1.15 s
  assert.equal(time('stone', null), 7500) // 7.5 s by hand, with no drop
  assert.equal(time('cobblestone', wooden), 1500) // about 1.5 s
  // About 2.25 s in the design: 2 / 3 / 30 per tick needs 45 ticks in float32 (vanilla), but
  // 1 / (2 / 3 / 30) is 45.000000000000014 in a double, so prismarine-block rounds up to 46.
  assert.equal(time('deepslate', wooden), 2300)
  assert(Math.ceil(1 / Math.fround(Math.fround(2 / 3) / 30)) === 45)
  assert.equal(time('cobbled_deepslate', wooden), 2650)
  assert.equal(time('blackstone', wooden), 1150)
  assert.equal(time('dirt', null), 750) // about 0.75 s
  assert.equal(time('grass_block', null), 900) // about 0.9 s
  // A wooden pickaxe is no faster on dirt than a hand.
  assert.equal(time('dirt', wooden), 750)
  // Off the ground: five times the breaking delta, rounded up to whole ticks.
  assert.equal(time('stone', wooden, false), 5650) // 4.91 times, not exactly 5
  assert.equal(time('cobblestone', wooden, false), 7500)
  assert.equal(time('deepslate', wooden, false), 11250)
  assert.equal(time('dirt', null, false), 3750)
  assert.equal(time('grass_block', null, false), 4500)
  assert.equal(time('stone', null, false), 37500)
  // The same through the digging plugin the bridge calls, with a held wooden pickaxe.
  const bot = new EventEmitter()
  const world = new Map()
  Object.assign(bot, {
    registry, heldItem: new Item(wooden, 1), inventory: { slots: [] }, getEquipmentDestSlot: () => 5, game: { gameMode: 'survival' },
    entity: { position: new Vec3(0.5, 64, 0.5), eyeHeight: 1.62, onGround: true, effects: {} },
    blockAt: position => world.get(`${Math.floor(position.x)},${Math.floor(position.y)},${Math.floor(position.z)}`) || block('air')
  })
  require(require.resolve('mineflayer/lib/plugins/digging', { paths: [mineflayerEntry] }))(bot)
  assert.equal(bot.digTime(block('stone')), 1150)
  assert.equal(bot.digTime(block('deepslate')), 2300)
  bot.entity.onGround = false
  assert.equal(bot.digTime(block('stone')), 5650)
  bot.entity.onGround = true
  bot.heldItem = null
  assert.equal(bot.digTime(block('dirt')), 750)
  assert.equal(bot.digTime(block('grass_block')), 900)
  // Eyes in water slow a dig five times as well.
  world.set('0,65,0', block('water'))
  assert.equal(bot.digTime(block('dirt')), 3750)
})
