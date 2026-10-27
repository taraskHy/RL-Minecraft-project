'use strict'
// Captures frames of a web page (for example prismarine-viewer showing the live bot) through
// the Chrome DevTools Protocol of an already running Chrome started with --remote-debugging-port.
//
//   node capture.js --debug-port 9333 --url http://localhost:3007 --out DIR [--wait 8] [--seconds 20] [--fps 10]
//
// Writes DIR/frame_00000.jpg, ... Frames are taken as fast as Chrome delivers them up to --fps.

const fs = require('fs')
const path = require('path')

const args = { debugPort: 9333, wait: 8, seconds: 20, fps: 10, width: 1280, height: 720 }
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = process.argv[i + 1]
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

async function main () {
  fs.mkdirSync(args.out, { recursive: true })
  const targets = await (await fetch(`http://127.0.0.1:${args.debugPort}/json/list`)).json()
  const page = targets.find(t => t.type === 'page')
  const ws = new WebSocket(page.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject })
  let id = 0
  const pending = new Map()
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data)
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) }
  }
  const send = (method, params = {}) => new Promise(resolve => { const k = ++id; pending.set(k, resolve); ws.send(JSON.stringify({ id: k, method, params })) })
  await send('Emulation.setDeviceMetricsOverride', { width: Number(args.width), height: Number(args.height), deviceScaleFactor: 1, mobile: false })
  await send('Page.navigate', { url: args.url })
  await sleep(Number(args.wait) * 1000)
  const end = Date.now() + Number(args.seconds) * 1000
  const step = 1000 / Number(args.fps)
  let n = 0
  while (Date.now() < end) {
    const t0 = Date.now()
    const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 85 })
    if (r.result && r.result.data) fs.writeFileSync(path.join(args.out, `frame_${String(n++).padStart(5, '0')}.jpg`), Buffer.from(r.result.data, 'base64'))
    const left = step - (Date.now() - t0)
    if (left > 0) await sleep(left)
  }
  ws.close()
  console.log(JSON.stringify({ frames: n, seconds: Number(args.seconds) }))
}

main().catch(err => { console.error(err); process.exit(1) })
