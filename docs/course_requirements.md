# Course requirements and project scope

The final goal is an agent that starts in a fresh Minecraft Survival world and defeats the Ender Dragon. A resource simulator or a prepared End fight is an intermediate development task, not completion of that goal. The project starts on 23 October 2026; the user supplied a deadline of 31 December 2026, a team of 3–4 students, and access to an RTX 5080 and 64 GB RAM. These are planning inputs, not evidence that the final task is already feasible or achieved.

Hardware clarification on 23 October: the next sessions use a laptop, followed by a 64 GB desktop whose NVIDIA model is still uncertain, and later the RTX 5080. The current simulator and local server use the CPU. See [machine setup](windows_setup.md#laptop-now-desktop-later) for the portable configuration.

## What governs the project

The final-project handout supplies the assessment requirements. The course outline and lecture decks supply learning material. The ideas document offers examples, and the NotebookLM-branded summary is a study aid. Instructions for UCL students embedded in the reused lecture slides are not the local project's submission instructions.

The user supplied the grading split: **final project 90%, class presentation 10%**. The provided final-project handout does not give finer numerical rubric weights. Its date, 1 October 2024, is a document date; it is not the submission deadline. No submission portal or AI-use policy was found in the supplied project handout.

Source page numbers below count PDF pages from 1, including title slides. Local source files are the user-provided originals in `C:/Users/ethan/Downloads/`.

## Required deliverables

| Requirement | Source | What we must produce |
| --- | --- | --- |
| Relevant, original/challenging, feasible project aligned with the course | `final_project.pdf`, p. 1 | A precise research question and an honest scope statement explaining the full Survival goal and the current implemented subset |
| Clearly define states, actions, rewards, dynamics, and objective | `final_project.pdf`, p. 1 | An environment specification for each experiment, including observations, terminal conditions, time limits, action semantics, and assumptions |
| Use available data or an existing/custom simulation with sufficient challenge | `final_project.pdf`, p. 1 | Reproducible environment setup and documented differences between simulation and the game |
| Justify algorithm choices using the problem characteristics and research | `final_project.pdf`, p. 2 | A comparison of why tabular Q-learning and SARSA suit the initial small state/action space, plus the limitations of that representation |
| Thorough hyperparameter exploration | `final_project.pdf`, p. 2 | A recorded search space, equal training budgets, validation-based selection, and a clear explanation of the selected settings |
| Define metrics and compare approaches | `final_project.pdf`, p. 2 | Random baseline, Q-learning/SARSA comparison, reward comparison, repeated runs, and uncertainty reporting |
| Written report: overview, formulation, methodology/tuning, results, discussion, conclusion | `final_project.pdf`, p. 2 | Quantitative and qualitative evidence, related-work comparison, limitations, and conclusions supported by the actual runs |
| Clean, readable, commented code, reproducible instructions, and version control | `final_project.pdf`, p. 2 | Organized source, setup/run commands, saved configurations/seeds, and Git history |
| Presentation of key findings; visual aids chosen by the team | `final_project.pdf`, p. 2 | A **15-minute** presentation with a demonstration and clear division of speaking roles |
| Formulation, algorithm appropriateness, and implementation quality are named evaluation criteria | `final_project.pdf`, pp. 2–3 | Explain the design and learning updates, verify behavior, and make the code maintainable |
| At most four students; all contribute; larger groups produce larger projects/reports | `final_project.pdf`, p. 3 | A truthful contribution record, with each member able to explain the full approach |
| Report length and formatting | `final_project.pdf`, p. 3 | **8–20 pages excluding references**, **Arial 12**, **1.5 line spacing**, **one-inch margins** |
| Academic integrity; no plagiarism or collusion | `final_project.pdf`, p. 3 | Attribute external material and document actual human and AI contributions without inventing work |

The concrete comparison and evaluation methods in the third column are our implementation plan. The handout requires justification, tuning, and evaluation; it does not prescribe these exact algorithms or experiment counts.

## Final demonstration contract

The requested outcome is **fresh-world Survival through an Ender Dragon defeat in actual Minecraft**. Before final evaluation, record the Minecraft edition/version, difficulty, world-generation rules, permitted observations, action interface, death/restart policy, and time budget. The demo must disclose its seed and starting conditions. A selected seed can support a demonstration; it cannot establish generalization to new seeds.

The implementation may use scripted low-level operations such as movement or crafting if that design is agreed with the instructor. Record which decisions are learned, which actions are scripted, and which observations come from the game API. Confirm that this hybrid scope satisfies the course's expectations early. This is a project-scoping recommendation, not a new permission requirement for developing the code.

Do not present any of the following as a completed fresh-world run: simulator success; a prepared inventory; teleportation to the End; a preconstructed portal; spawning or directly damaging the dragon by command; or a fully scripted strategy described as learned. Prepared encounters are useful training tests when they are labeled explicitly. Final evidence should include the uninterrupted run or complete logs/video, checkpoint/configuration, and the actual dragon-defeat event.

The initial implementation is a learning foundation: a small resource/crafting simulator, tabular RL, controlled experiments, and an interface toward real-game tasks. The final goal remains open until actual full-run evidence exists. Hardware availability alone does not establish that full Survival can be learned within the deadline.

## Planned research questions

1. How do Q-learning and SARSA differ in completion rate and sample efficiency on the same resource/crafting task?
2. Does informative reward feedback improve learning relative to sparse completion rewards, and does it create unintended behavior?
3. Which state features and action abstractions transfer from the simulator to comparable real-game subtasks?
4. When learned decisions are embedded in a hybrid controller, what do they add over a scripted or random decision baseline?

These questions can be evaluated before full Survival succeeds. A strong answer to them is useful project evidence, but does not replace the requested final demonstration.

## Proposed ten-week schedule

These are future targets, not completed results. There are 69 days between 23 October and 31 December 2026, approximately ten weeks.

| Dates in 2026 | Planned evidence and decision |
| --- | --- |
| 23–29 Oct | Everyone explains the MDP and updates; run a reproducible baseline; record game version/interface and final-demo rules; clarify hybrid scope with the instructor |
| 30 Oct–5 Nov | Validate resource/crafting simulator; compare random, Q-learning, and SARSA; test real-game observation, reset, and action interfaces |
| 6–12 Nov | Demonstrate real-world resource collection/crafting; test reward variants and exploration schedules; record failure cases |
| 13–19 Nov | Integrate early Survival progression and survival management; measure success and bottlenecks on independent starts |
| 20–26 Nov | Work on Nether progression and required resources; use prepared training scenarios only as clearly labeled development aids |
| 27 Nov–3 Dec | Work on stronghold/portal progression and End combat; assess whether the full pipeline can be integrated in time |
| 4–10 Dec | Attempt complete fresh-world runs; freeze representation and algorithm choices unless a measured failure requires change |
| 11–17 Dec | Evaluate repeated runs, document actual completion/failure rates, finish ablations and related-work comparison |
| 18–24 Dec | Write and format the report; prepare plots, demonstration evidence, and the 15-minute presentation |
| 25–31 Dec | Reproduce results from clean instructions, rehearse, verify contributions/citations, and submit by the user-supplied deadline |

At each gate, record what works, what failed, and what changes next. If the full-run goal becomes infeasible, raise that fact promptly with evidence rather than quietly relabeling a smaller task as the requested result.

## Evaluation and reproducibility plan

- Separate training, validation, and final testing. Select hyperparameters on validation outcomes; do not repeatedly tune to the final test seeds.
- Use multiple independently trained agents, not only many evaluation episodes from one checkpoint. A practical initial plan is at least five training seeds for cheap simulator experiments, adjusted transparently for measured runtime.
- Give algorithms comparable interaction budgets. Record transitions, episodes, elapsed time, hardware, and all selected hyperparameters.
- Report task success, episode length, return, deaths/failures, and learning curves. Report original task reward separately from any shaped training reward; returns from different reward definitions are not directly comparable.
- Show uncertainty across training seeds and retain failed runs. For successful-episode length, state the number of successes so failed episodes do not disappear from the comparison.
- Keep simulator results, prepared real-game encounters, and complete fresh-world Survival results in separate tables with explicit environment labels.
- Save configuration, seeds, learned checkpoint, code revision when available, and an outcome log. A favorable demonstration alone is not a reliable success-rate estimate.

## Contribution and AI assistance record

Every member should learn the core material, review another member's work, and contribute to the report/presentation. Suggested responsibilities for a four-person team are environment/game integration, RL algorithms, experiments/analysis, and reproducibility/report/demo. For three people, combine the last two while sharing reviews. These are proposed responsibilities; no names, assignments, or completed contributions are asserted here.

Maintain a real record with date, contributor, work performed, artifact/commit, verification, and reviewer. Keep an analogous AI-assistance log describing generated code/text, suggestions adopted, and human verification. This repository was started with AI assistance reviewing the supplied course material and drafting project scaffolding/documentation. That fact does not establish what each student has understood or verified. The provided handout states general academic-integrity obligations, not a specific AI policy; follow any additional policy communicated by the instructor and do not fabricate one.

## Inventory of all supplied material

The lecture titles below come from the actual decks. Their numbering does **not** match the lecture-topic numbering in `course_structure.pdf`.

| Supplied material | Relevant content and pages | Role in this project |
| --- | --- | --- |
| `ideas (1).docx` | Eight example projects: pricing, warehouse paths, queues, traffic signals, Pong, forest fire, elevator, garden; no binding rubric | Examples of manageable environments and simple RL methods, not an instruction to choose one |
| `course_summary.pdf` | MDP/Bellman p. 3; exploration p. 4; MC/SARSA/Q-learning p. 5; approximation/DQN pp. 6–8; policy gradients/PPO/GAE pp. 9–11; imitation pp. 12–14 | Visual revision aid; not a replacement for the technical decks or the rubric |
| `final_project.pdf` | Requirements p. 1; methodology/deliverables p. 2; groups, integrity, report format p. 3 | Authoritative project specification supplied by the user |
| `course_structure.pdf` | Course topics p. 1; simulation, comparison/tuning, and debugging practical sessions p. 2 | Overall course scope; does not require implementing every listed topic |
| `1.pdf` | Introduction; interaction/reward/value pp. 21–26; state and partial observability pp. 29–40; policy/value/model pp. 43–51 | Foundation for the problem formulation and state limitations |
| `2.pdf` | Exploration and Exploitation; tradeoff p. 6; greedy/epsilon-greedy pp. 15–20; advanced exploration later in the deck | Core exploration strategy; advanced bandit methods are optional background |
| `3.pdf` | Markov Decision Processes and Dynamic Programming; MDP pp. 6–11; returns/policies/values pp. 16–28; planning pp. 38–64 | Formal specification and Bellman reasoning; exact planning is not required for Minecraft |
| `5.pdf` | Model-Free Prediction; MC pp. 19–25; TD pp. 26–41; multi-step returns/traces pp. 53–79 | Understand bootstrapping and the one-step update before implementing control |
| `6.pdf` | Model-Free Control; SARSA pp. 16–19; off-policy/Q-learning pp. 20–27; targets/convergence pp. 48–49 | Main implementation and comparison; Double Q and importance sampling remain optional |
| `7.pdf` | Function Approximation in Reinforcement Learning; approximation pp. 4–35; instability pp. 36–51; replay/DQN pp. 57–62 | Explains limits of tables and a conditional DQN extension |
| `8.pdf` | Planning and Models; inaccurate models pp. 26–28; Dyna pp. 29–36; search pp. 45–59 | Understand simulator mismatch; Dyna/MCTS are not part of the initial algorithm scope |
| `9.pdf` | Policy Gradients and Actor Critics; tradeoffs pp. 6–8; gradients pp. 19–36; actor-critic pp. 37–42 | Comparative background; not a prerequisite for the core tabular implementation |
| `14.pdf` | Deep Reinforcement Learning Part 2; general value functions pp. 4–18; multi-task/normalization pp. 19–30; distributional RL pp. 31–35 | Advanced context to defer; no requirement to implement these topics |
