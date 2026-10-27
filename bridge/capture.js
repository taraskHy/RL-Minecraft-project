'use strict'
// Captures frames of a web page (for example prismarine-viewer showing the live bot) through
// the Chrome DevTools Protocol of an already running Chrome started with --remote-debugging-port.
//
//   node capture.js --debug-port 9333 --url http://localhost:3007 --out DIR [--wait 8] [--seconds 20] [--fps 10]
//                   [--screencast true] [--stop-file PATH] [--caption TEXT]
//
// Writes DIR/frame_00000.jpg, ... Frames are taken as fast as Chrome delivers them up to --fps.
// With --screencast, Chrome streams every frame it paints instead (smoother), and DIR/frames.json
// lists each frame's wall-clock time in milliseconds so a video can be cut to exact moments.
// Capture stops after --seconds or as soon as --stop-file exists.

const fs = require('fs')
const path = require('path')

const args = { debugPort: 9333, wait: 8, seconds: 20, fps: 10, width: 1280, height: 720, screencast: 'false', stopFile: '', caption: '' }
for (let i = 2; i < process.argv.length; i += 2) {
  args[process.argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = process.argv[i + 1]
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
const stopped = () => args.stopFile && fs.existsSync(args.stopFile)

async function main () {
  fs.mkdirSync(args.out, { recursive: true })
  const targets = await (await fetch(`http://127.0.0.1:${args.debugPort}/json/list`)).json()
  const page = targets.find(t => t.type === 'page')
  const ws = new WebSocket(page.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject })
  let id = 0
  const pending = new Map()
  const events = []
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data)
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) } else if (m.method) events.forEach(f => f(m))
  }
  const send = (method, params = {}) => new Promise(resolve => { const k = ++id; pending.set(k, resolve); ws.send(JSON.stringify({ id: k, method, params })) })
  await send('Emulation.setDeviceMetricsOverride', { width: Number(args.width), height: Number(args.height), deviceScaleFactor: 1, mobile: false })
  await send('Page.navigate', { url: args.url })
  await sleep(Number(args.wait) * 1000)
  if (args.caption) {
    const css = 'position:fixed;left:24px;bottom:22px;padding:10px 18px;border-radius:8px;background:rgba(10,16,22,.6);' +
      'color:#fff;font:600 26px system-ui,sans-serif;z-index:9;pointer-events:none'
    await send('Runtime.evaluate', { expression: `(() => { const d = document.createElement('div'); d.style.cssText = ${JSON.stringify(css)}; d.textContent = ${JSON.stringify(args.caption)}; document.body.appendChild(d) })()` })
  }
  const end = Date.now() + Number(args.seconds) * 1000
  let n = 0
  if (args.screencast === 'true') {
    const times = []
    events.push(m => {
      if (m.method !== 'Page.screencastFrame') return
      const { data, metadata, sessionId } = m.params
      const file = `frame_${String(n++).padStart(5, '0')}.jpg`
      fs.writeFileSync(path.join(args.out, file), Buffer.from(data, 'base64'))
      times.push({ file, t: Math.round(metadata.timestamp * 1000) })
      send('Page.screencastFrameAck', { sessionId })
    })
    await send('Page.startScreencast', { format: 'jpeg', quality: 90, maxWidth: Number(args.width), maxHeight: Number(args.height), everyNthFrame: 1 })
    while (Date.now() < end && !stopped()) await sleep(100)
    await send('Page.stopScreencast')
    fs.writeFileSync(path.join(args.out, 'frames.json'), JSON.stringify(times))
  } else {
    const step = 1000 / Number(args.fps)
    while (Date.now() < end && !stopped()) {
      const t0 = Date.now()
      const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 85 })
      if (r.result && r.result.data) fs.writeFileSync(path.join(args.out, `frame_${String(n++).padStart(5, '0')}.jpg`), Buffer.from(r.result.data, 'base64'))
      const left = step - (Date.now() - t0)
      if (left > 0) await sleep(left)
    }
  }
  ws.close()
  console.log(JSON.stringify({ frames: n, seconds: Number(args.seconds) }))
}

main().catch(err => { console.error(err); process.exit(1) })
