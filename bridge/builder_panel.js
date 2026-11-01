'use strict'
// The course builder's control panel (bridge/builder_bot.js), for a player who cannot chat (chat turned off in the
// game's options or by the account): a row of command blocks just north of the spawn pad, each with a button and
// a sign. A button makes its command block send the builder bot a system message, "builder:COMMAND[:OPTION] PLAYERS",
// PLAYERS being the players (not spectators) within reach of the button; the bot acts as on that player's chat
// command (!COMMAND OPTION) when there is one, and asks for one at a time when there are several.
// The builder runs its server with command blocks on (parkour/build_session.py); its scans leave these cells out.

const PAD = [-20, 100, -20] // parkour/server.py SPAWN_PAD: players appear on its 5x5 stone floor at y - 1
const ROW_Z = PAD[2] - 3 // the command blocks stand just north of the floor (whose cells run from z - 2 to z + 2)
const ROW_Y = PAD[1] + 1 // at eye height for a player on the pad
const BUTTONS = [ // [the sign's two lines, the command (and option) the button sends]
  ['Kit', 'creative', 'build'],
  ['Route', 'what it sees', 'route'],
  ['Run', 'router', 'run'],
  ['Run', 'course list', 'run:list'],
  ['Demo', 'next course', 'demo'],
  ['Clear', 'marks', 'clear'],
  ['Help', '', 'help']
]
const X0 = PAD[0] - Math.floor(BUTTONS.length / 2) // the row is centred on the pad

// Button i's command block, the button on its south face and the sign on top. A powered command block also powers
// the command blocks touching it, so the blocks zig-zag (odd ones a block higher) and no two share a face.
function place (i) {
  const x = X0 + i; const y = ROW_Y + (i % 2)
  return { block: [x, y, ROW_Z], button: [x, y, ROW_Z + 1], sign: [x, y + 1, ROW_Z] }
}

// [x, y, z, what] of every block the panel sets.
function cells () {
  return BUTTONS.flatMap((_, i) => {
    const p = place(i)
    return [[...p.block, 'command_block'], [...p.button, 'button'], [...p.sign, 'sign']]
  })
}

// The command a button runs: a system message to the bot naming the players within reach of it ("A, B" for two),
// the bot and spectators (who cannot press buttons) left out.
function tellraw (botName, command) {
  return `tellraw ${botName} ["builder:${command} ",{"selector":"@a[distance=..7,gamemode=!spectator,name=!${botName}]"}]`
}

// The panel's box emptied first, so no block of an older layout touches a new command block: the buttons' row,
// then the rest from the top down, so no button or sign is ever left without its block (it would drop as an item).
function clearing () {
  const x0 = X0 - 1; const x1 = X0 + BUTTONS.length
  const out = [`/fill ${x0} ${ROW_Y - 1} ${ROW_Z + 1} ${x1} ${ROW_Y + 2} ${ROW_Z + 1} minecraft:air replace`]
  for (let y = ROW_Y + 2; y >= ROW_Y - 1; y--) out.push(`/fill ${x0} ${y} ${ROW_Z} ${x1} ${y} ${ROW_Z} minecraft:air replace`)
  return out
}

// The commands that build the panel: the clearing, then /setblock for the command blocks (a button needs its
// block), the buttons and the signs, each under the 256 characters a chat command may have. Minecraft 1.21.4: text
// in JSON, sign text in front_text.messages.
function commands (botName) {
  const out = clearing()
  out.push(...BUTTONS.map(([, , command], i) =>
    `/setblock ${place(i).block.join(' ')} minecraft:command_block{Command:'${tellraw(botName, command)}'} replace`))
  BUTTONS.forEach(([line1, line2], i) => {
    out.push(`/setblock ${place(i).button.join(' ')} minecraft:stone_button[face=wall,facing=south] replace`)
    out.push(`/setblock ${place(i).sign.join(' ')} minecraft:oak_sign[rotation=0]{front_text:{messages:` +
      `['${JSON.stringify(line1)}','${JSON.stringify(line2)}','""','""']},is_waxed:1b} replace`)
  })
  return out
}

// "builder:run:list Ethan" -> { cmd: '!run', opt: 'list', players: ['Ethan'] }; null for any other message.
// players is empty when nobody was near the button.
function parse (message) {
  const m = /^builder:([a-z]+)(?::([a-z0-9]+))?(?: ([A-Za-z0-9_]{1,16}(?:, [A-Za-z0-9_]{1,16})*))?$/.exec(String(message).trim())
  return m ? { cmd: '!' + m[1], opt: m[2] || '', players: m[3] ? m[3].split(', ') : [] } : null
}

module.exports = { PAD, BUTTONS, cells, commands, parse, tellraw }
