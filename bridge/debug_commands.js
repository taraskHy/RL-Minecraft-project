'use strict'
// Debug helper: connect, send a few operator commands, and print what comes back.
//   node debug_commands.js PORT
const mineflayer = require('mineflayer')
const port = Number(process.argv[2] || 25570)
const bot = mineflayer.createBot({ host: '127.0.0.1', port, username: 'ParkourBot', version: '1.21.4', auth: 'offline' })
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
bot.on('messagestr', (m, pos) => console.log('[msg]', pos, m))
bot._client.on('position', p => console.log('[position packet]', JSON.stringify(p)))
bot.once('spawn', async () => {
  console.log('spawned at', bot.entity.position, 'yaw', bot.entity.yaw)
  await sleep(1000)
  for (const cmd of ['/fill 4 95 44 6 95 46 minecraft:stone', '/tp ParkourBot 5.5 96 44.5 90 0']) {
    console.log('sending', cmd)
    bot.chat(cmd)
    await sleep(1500)
    console.log('position now', bot.entity.position, 'block below', bot.blockAt(bot.entity.position.offset(0, -0.5, 0))?.name)
  }
  console.log('attributes', JSON.stringify(bot.entity.attributes))
  bot.quit()
  await sleep(300)
  process.exit(0)
})
bot.on('kicked', r => { console.log('kicked', JSON.stringify(r)); process.exit(1) })
bot.on('error', e => console.log('error', e.message))
