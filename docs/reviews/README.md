# Review records

- `stone_s1_round1_findings.json`: the first adversarial review of stone step S1
  (commit `c441c5f`, 24 October 2026). It lists 49 findings from seven review
  dimensions. Each has the reviewer's claim (`failure_scenario`, `evidence`,
  `suggested_fix`) and an independent verifier's verdict (`verification`:
  `CONFIRMED`, `PLAUSIBLE` or `REFUTED`, with `fix_notes`). 37 were confirmed
  and fixed in the next commit on `codex/stone-tools`; 12 were refuted. Paths
  it mentions under a `scratchpad` folder were on the old PC and are gone.
- `evidence/light_update_jarscan.py` and `evidence/light_update_jarscan_players.py`:
  read-only bytecode scans of the vanilla 1.21.4 server jar behind decision 9's
  fallback. They show that the server builds its light-update packet only in
  `ChunkHolder.broadcastChanges` and sends it only to players for whom the chunk
  lies on the edge of their view, so the bot never receives light for its own
  digs. After `setup.cmd`, run from the repository root:
  `.venv\Scripts\python.exe docs\reviews\evidence\light_update_jarscan_players.py`.
  The obfuscated names (`aqg`, `aqi`, `adb`, `dfp`) are those of the 1.21.4 jar.
