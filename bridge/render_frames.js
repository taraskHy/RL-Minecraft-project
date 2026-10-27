'use strict'
// Renders a page that exposes window.renderAt(tick) frame by frame and saves screenshots,
// through the DevTools Protocol of a running Chrome (--remote-debugging-port).
//
//   node render_frames.js --debug-port 9333 --url "file:///.../theatre.html?render=1&race=0" --out DIR
//                         [--fps 30] [--speed 1] [--hold 2] [--width 1920] [--height 1080]
//
// Time is stepped exactly (20 game ticks per second at speed 1), so the video is smooth
// however slowly the page renders.

const fs = require('fs')
const path = require('path')

const args = { debugPort: 9333, fps: 30, speed: 1, hold: 2, width: 1920, height: 1080 }
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = process.argv[i + 1]
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

async function main () {
  fs.mkdirSync(args.out, { recursive: true })
  const page = (await (await fetch(`http://127.0.0.1:${args.debugPort}/json/list`)).json()).find(t => t.type === 'page')
  const ws = new WebSocket(page.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject })
  let id = 0
  const pending = new Map()
  ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) } }
  const send = (method, params = {}) => new Promise(resolve => { const k = ++id; pending.set(k, resolve); ws.send(JSON.stringify({ id: k, method, params })) })
  const evaluate = async expr => (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result.result.value
  await send('Emulation.setDeviceMetricsOverride', { width: Number(args.width), height: Number(args.height), deviceScaleFactor: 1, mobile: false })
  await send('Page.navigate', { url: args.url })
  for (let k = 0; k < 600; k++) { if (await evaluate('window.renderReady === true')) break; await sleep(100) }
  await sleep(1500) // fonts and textures
  const info = await evaluate('JSON.stringify(window.renderInfo())')
  const { maxTicks } = JSON.parse(info)
  const ticksPerFrame = 20 * Number(args.speed) / Number(args.fps)
  const frames = Math.ceil(maxTicks / ticksPerFrame) + Math.round(Number(args.hold) * Number(args.fps))
  for (let f = 0; f < frames; f++) {
    await evaluate(`window.renderAt(${Math.min(maxTicks, f * ticksPerFrame)})`)
    const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 92 })
    fs.writeFileSync(path.join(args.out, `frame_${String(f).padStart(5, '0')}.jpg`), Buffer.from(r.result.data, 'base64'))
  }
  ws.close()
  console.log(JSON.stringify({ frames, maxTicks, info: JSON.parse(info) }))
}

main().catch(err => { console.error(err); process.exit(1) })
