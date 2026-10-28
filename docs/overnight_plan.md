# Overnight plan, 28 October 2026 (decision tree)

Written at 01:15 before the user went to sleep. Each branch says what the
assistant does depending on what the runs show. Every outcome is recorded in
TEAM_STATUS.md and pushed.

```mermaid
flowchart TD
    S[01:15 runs in progress] --> A[Neo training advanced-ft3 ends ~02:30]
    S --> B[Double-neo search with set-back walls ends ~01:30]
    S --> V[Visuals, all night, beside the runs]

    A --> A0{Level 6 test >= 85% and level 7 >= 50%?}
    A0 -- yes --> A1[Test on unseen neo courses: simulator, offline bot, real Minecraft]
    A0 -- no, still improving --> A2[Train 1500 more iterations, learning-rate cap 3e-4]
    A0 -- no, collapsed or flat --> A3[Restart from best.pt, cap 3e-4, entropy 0.01]
    A2 --> A0b{Better now?}
    A3 --> A0b
    A0b -- yes --> A1
    A0b -- no --> A4[Keep the best neo policy; report the level reached]
    A1 --> A5{Standard and obstacle tests within 2 points of the final policy?}
    A5 -- yes --> A6[Promote it: one final policy for everything]
    A5 -- no --> A7[Keep two policies: final for normal courses, neo policy for levels 6-8]

    B --> B0{Any double neo landed?}
    B0 -- yes --> B1[Add to levels 7-8, fine-tune 1000 iterations, test]
    B0 -- no --> B2[Report double and triple neos as impossible in Minecraft physics, with the search as evidence; long single neos stay]

    A6 --> C[Hyperparameter study, 3 runs at a time ~3 h]
    A7 --> C
    A4 --> C
    C --> C0{A setting beats the defaults by > 3 points on validation?}
    C0 -- yes, and time left --> C1[Train a final run with it; keep it only if the test confirms]
    C0 -- no --> C2[Report the table; defaults stay]

    V --> V1[Minecraft-style 3D viewer: textured blocks, player model, sky, shadows]
    V --> V2[Ghost race: every training stage runs the same course at once]
    V --> V3[Colourful real-Minecraft showcase course, played by the best policy]
    V --> V4[Presentation dashboard of the key results]

    C1 --> M[Morning: results, TEAM_STATUS, report draft, everything pushed]
    C2 --> M
    B1 --> M
    B2 --> M
    V4 --> M
```

## Rules for the night

- Never more than three training processes on the GPU at once (more spills
  into system RAM and everything slows 5 to 10 times).
- Every claim keeps saying where it was measured: simulator, offline bot
  (Mineflayer's physics without a server) or real Minecraft.
- Visuals never change the physics or the policy. Colourful blocks in real
  Minecraft are full cubes with normal friction only (concrete, quartz, gold,
  emerald), so the bot moves exactly as on stone.
- The previous final policy stays available whatever happens.

## Update, 02:40: two new branches (requested from the side chat)

```mermaid
flowchart TD
    P[Greedy route planner: the existing policy, no course list] --> P1[Tune on validation courses, test on 1,200 + 1,200 courses]
    P1 --> P2[Same plans in Python and the bot: parity test]
    P2 --> P3[Offline bot and real Minecraft, standard and obstacle courses, replays]
    P3 --> N[Learned no-route model: wide scan + landing memory, route reward only in training]
    N --> N0{Beats the planner on validation?}
    N0 -- yes --> N1[It is the new-map model: offline, real Minecraft, replays, video]
    N0 -- no --> N2[Planner is the new-map model; the learned one is the research comparison]
```

Rule change: at most two training processes on the GPU (a third now spills
out of GPU memory and slows all of them about 8 times).
