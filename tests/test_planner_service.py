"""Planner failures must resolve requests, while normal shutdown stays quiet."""
import json
import subprocess

import pytest

from parkour.paths import BRIDGE, NODE


@pytest.mark.skipif(not NODE.exists(), reason="portable Node runtime not installed (run setup.cmd)")
def test_service_failure_and_clean_shutdown():
    script = r"""
const assert = require('assert/strict')
const { EventEmitter } = require('events')
const { PassThrough } = require('stream')
const children = []
require('child_process').spawn = () => {
  const child = new EventEmitter()
  child.stdin = new PassThrough()
  child.stdout = new PassThrough()
  children.push(child)
  return child
}
const { PlannerService } = require('./assist')
const warnings = []
process.stderr.write = text => { warnings.push(text); return true }
;(async () => {
  const normal = new PlannerService('fake', [], '.')
  const request = normal.plan({ index: 0 })
  children[0].stdout.write('{"plan":[[1,1,0,0,3]]}\n')
  assert.deepEqual(await request, { plan: [[1,1,0,0,3]] })
  normal.close()
  children[0].emit('exit', 0)
  assert.equal(warnings.length, 0)

  const failed = new PlannerService('fake', [], '.')
  const pending = failed.plan({ index: 1 })
  children[1].emit('exit', 1)
  assert.deepEqual(await pending, { plan: null, error: 'planner exited (1)' })
  assert.deepEqual(await failed.plan({ index: 2 }), { plan: null, error: 'planner exited (1)' })
  children[1].stdin.emit('error', new Error('closed'))
  assert.equal(warnings.length, 1)

  const interrupted = new PlannerService('fake', [], '.')
  const unanswered = interrupted.plan({ index: 3 })
  interrupted.close()
  children[2].emit('exit', 0)
  assert.equal((await unanswered).error, 'planner exited (0)')
  assert.equal(warnings.length, 2) // close with an unanswered request is still a failure
  console.log('planner lifecycle checks completed')
})().catch(error => { console.log(error.stack); process.exitCode = 1 })
"""
    result = subprocess.run([str(NODE), "-e", script], cwd=BRIDGE, check=True,
                            capture_output=True, text=True, timeout=15)
    assert 'planner lifecycle checks completed' in result.stdout


def test_report_marks_degraded_assist(tmp_path):
    from parkour.report import game_row
    from parkour.handoff import games_lines

    folder = tmp_path / 'games' / 'failed-assist'
    folder.mkdir(parents=True)
    summary = dict(mode='offline', episodes=[dict(result='success')],
                   sim_success_same_courses=1, outcome_agreement=1, replay_max_pos_err=0,
                   assist=dict(planner_errors=2))
    (folder / 'summary.json').write_text(json.dumps(summary))
    row = game_row(tmp_path, 'failed-assist')
    assert 'degraded assist: 2 planner errors' in row
    assert row.split('|')[6].strip() == '-'  # no server exists in an offline run
    assert any('degraded assist: 2 planner errors' in line for line in games_lines(tmp_path))
