'use strict'

// Connection/observation infrastructure only. No policy, chat commands, resets,
// inventory changes, movement commands, or privileged world queries.
function parseArgs (args) {
  if (args.length !== 2) throw new Error('Usage: node survival-probe.js PORT SECONDS')
  const [port, seconds] = args.map(Number)
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid port')
  if (!Number.isInteger(seconds) || seconds < 1 || seconds > 120) throw new Error('Seconds must be 1..120')
  return { port, seconds }
}

function observationReady (bot) {
  const p = bot.entity?.position
  return Number.isFinite(bot.health) && Number.isFinite(bot.food) &&
    Number.isFinite(p?.x) && Number.isFinite(p?.y) && Number.isFinite(p?.z)
}

function observation (bot) {
  if (!observationReady(bot)) {
    throw new Error('Observation is not ready: health, food, and all position coordinates must be finite')
  }
  const p = bot.entity.position
  return {
    timestamp: new Date().toISOString(),
    evidence: 'Minecraft Survival connection only; no learned behavior',
    position: { x: p.x, y: p.y, z: p.z },
    health: bot.health,
    food: bot.food,
    dimension: bot.game.dimension,
    gamemode: bot.game.gameMode,
    inventory: bot.inventory.items().map(item => ({ name: item.name, count: item.count })),
    dead: bot.health <= 0
  }
}

function main () {
  const { port, seconds } = parseArgs(process.argv.slice(2))
  const mineflayer = require('mineflayer')
  const bot = mineflayer.createBot({
    host: '127.0.0.1', port, username: 'SurvivalProbe', version: '1.21.4',
    auth: 'offline', respawn: false, hideErrors: true
  })
  let finished = false
  let interval
  let observationCount = 0
  let deadline = setTimeout(() => finish(new Error('Timed out waiting for a Survival spawn')), 45000)

  function finish (error) {
    if (finished) return
    finished = true
    clearInterval(interval)
    clearTimeout(deadline)
    bot.removeListener('health', record)
    if (!error && observationCount === 0) {
      error = new Error('The observation deadline passed without complete health, food, and position data')
    }
    if (error) {
      process.stderr.write(`[survival probe] ${error.message}\n`)
      process.exitCode = 2
    } else {
      process.stdout.write(JSON.stringify({ event: 'connection_check_complete', observations: observationCount }) + '\n')
    }
    bot.quit('Local observation check finished')
    // A lost connection must not leave the launcher waiting forever.
    setTimeout(() => process.exit(process.exitCode || 0), 2000).unref()
  }

  function record () {
    if (finished || !observationReady(bot)) return
    try {
      process.stdout.write(JSON.stringify(observation(bot)) + '\n')
      observationCount++
    } catch (error) {
      finish(error)
    }
  }

  bot.once('spawn', () => {
    if (finished) return
    clearTimeout(deadline)
    if (bot.game.gameMode !== 'survival') {
      finish(new Error(`Expected survival gamemode, observed ${bot.game.gameMode}`))
      return
    }
    record()
    if (finished) return
    // Mineflayer can emit spawn before assigning the first health packet's
    // values. Record that packet once it is applied, without inventing data.
    if (!observationReady(bot)) bot.once('health', record)
    interval = setInterval(record, 1000)
    deadline = setTimeout(() => finish(), seconds * 1000)
  })
  bot.once('death', () => finish(new Error('Bot died in the ordinary Survival world; the check did not respawn or reset it')))
  bot.once('error', error => finish(error))
  bot.once('kicked', reason => finish(new Error(`Kicked: ${JSON.stringify(reason)}`)))
  bot.once('end', reason => {
    if (!finished) finish(new Error(`Unexpected disconnect: ${reason}`))
  })
  process.once('SIGINT', () => finish(new Error('Interrupted')))
  process.once('SIGTERM', () => finish(new Error('Stopped')))
}

if (require.main === module) {
  try { main() } catch (error) {
    process.stderr.write(`[survival probe] ${error.message}\n`)
    process.exitCode = 2
  }
}

module.exports = { parseArgs, observationReady, observation }
