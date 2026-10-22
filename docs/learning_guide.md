# Learning guide for the Minecraft RL project

The team should be able to derive the agent's update, explain its state and rewards, and reproduce its results. Start with a small resource/crafting task so mistakes are visible and experiments are inexpensive. The requested final outcome remains a fresh Minecraft Survival run ending in an Ender Dragon defeat; success in the teaching simulator is only an intermediate result.

The reading references use the supplied filenames and one-based PDF page numbers. The complete material inventory and assessment requirements are in [course_requirements.md](course_requirements.md). Learn the first six sections together before dividing implementation work.

## 1. Describe one decision as an MDP

Read `1.pdf` pp. 21–40 and `3.pdf` pp. 6–20.

An agent receives a state or observation, chooses an action, receives a reward, and reaches a new situation. An MDP consists of states `S`, actions `A`, transition probabilities `P`, rewards `R`, and discount factor `gamma`. A policy maps states to actions or action probabilities. The objective is expected cumulative reward, not merely the next reward.

For a small crafting simulator, a state might contain position, resource availability, inventory, and crafted equipment. Actions could be movement, gathering, and crafting. The actual fields and legal actions must be documented from the implementation rather than inferred from the game's theme.

The Markov assumption means the current state contains everything needed to predict the next-state and reward distribution given the action. A screenshot or a few coarse inventory features are not automatically Markov. If two situations with identical encoded state behave differently because of omitted health, orientation, cooldown, or resource availability, explain this approximation and measure its effects. A finite-horizon task may need remaining time in the state.

**Exercise:** Write five real transitions from the simulator: `(state, action, reward, next_state, terminal)`. Explain what changes and what does not. Construct a pair of situations that expose a missing state feature.

## 2. Understand return and value

Read `3.pdf` pp. 16–28 and `5.pdf` pp. 26–41.

The discounted return is:

```text
G_t = r_(t+1) + gamma*r_(t+2) + gamma^2*r_(t+3) + ...
```

`Q(s, a)` estimates how much future return follows taking action `a` in state `s`, under the continuation policy being learned. A positive immediate reward is not necessarily a good decision if it leads to failure later. `gamma` controls the relative weight of later rewards; it is a modeling and learning choice, not a probability of success.

Temporal-difference learning updates an estimate using one observed reward and another current estimate. This is bootstrapping. Monte Carlo waits for the episode's observed return. One-step TD is the only update horizon needed for the initial implementation.

**Exercise:** Calculate the return for rewards `[0, 0, 10]` at `gamma = 0.9`. Then explain why rewarding repeated movement could compete with the intended crafting objective.

## 3. Explore with epsilon-greedy actions

Read `2.pdf` pp. 6, 15–20 and `6.pdf` pp. 12–13.

With probability `epsilon`, choose a random available action. Otherwise choose an action with the highest estimated Q-value. Random tie-breaking avoids always favoring the first action in a list. A decreasing exploration schedule can help the agent first gather information and later exploit what it learned.

Record the initial epsilon, final epsilon, and decay schedule. If actions are masked, explain what the mask knows and use the same rules across comparisons. A mask that reveals hidden world information changes the task.

For greedy evaluation, set epsilon to zero and stop learning; document tie-breaking and any other evaluation randomness. A practical finite training run with fixed learning rates or an epsilon floor is not covered automatically by an asymptotic convergence theorem.

**Exercise:** For `Q(s) = [1, 4, 2]` and epsilon `0.2`, calculate each action's probability when random exploration samples all three actions uniformly.

## 4. Implement and explain Q-learning

Read `6.pdf` pp. 20–27 and 48.

```text
target = reward                         if next state is terminal
target = reward + gamma*max_a Q(s', a)  otherwise
Q(s, a) = Q(s, a) + alpha*(target - Q(s, a))
```

`alpha` controls the update size. The error `target - Q(s, a)` says how surprising the transition was. Q-learning uses the best estimated next action in its target, even when exploration selected a different behavior action; this is why it is off-policy.

For example, with `Q(s,a)=2`, reward `1`, gamma `0.9`, best next Q-value `5`, and alpha `0.1`, the target is `5.5` and the new Q-value is `2.35`. If that transition ends the task, the target is `1` and the new value is `1.9`.

A real terminal transition has no future reward. A training rollout cut short by an external limit is different: whether to bootstrap depends on the task definition. Define whether the horizon is part of the task, and keep time-limit handling consistent in the environment, learner, and report.

**Exercise:** Work the numerical update by hand, locate it in the code, and explain how an accidental terminal bootstrap would distort learning.

## 5. Compare with SARSA

Read `6.pdf` pp. 16–19, 23, and 49.

SARSA uses the next action actually selected by the behavior policy:

```text
a' = epsilon_greedy(Q, s')
target = reward                       if next state is terminal
target = reward + gamma*Q(s', a')     otherwise
Q(s, a) = Q(s, a) + alpha*(target - Q(s, a))
```

For a nonterminal step, execute that selected `a'` next; do not silently select a second action and train from a different one. SARSA is on-policy because its update follows the policy generating experience. It can learn different behavior from Q-learning while exploration remains active. Neither algorithm guarantees safe exploration or successful Minecraft completion.

**Exercise:** Reuse the Q-learning example, but let the exploratory next action have value `-1`. Compare the two updates and explain the difference in plain language.

## 6. Evaluate what was learned

Read `final_project.pdf` p. 2 and the practical-session topics in `course_structure.pdf` p. 2.

Start with random actions, Q-learning, and SARSA under the same observation/action interface and training budget. Keep training seeds separate from validation and test seeds. Tune on validation outcomes and reserve final tests for the selected settings. Use multiple training seeds because one lucky trajectory or initial table is not reliable evidence.

Measure completion rate, episode length, failures, and task reward. Plot learning against environment transitions as well as episodes when episode lengths vary. Report wall-clock cost separately. Compare task reward across reward variants, not totals that mix incompatible shaped reward definitions. A recorded success helps explain behavior, but repeated evaluation estimates reliability.

**Exercise:** Explain why 100 episodes from one trained agent do not replace five independently trained agents. Identify one way a simulator can be solved without learning the intended behavior.

## 7. Optional reward shaping

Reward design is included in `course_structure.pdf` p. 1. Potential-based shaping is a proposed, small extension; the supplied materials do not establish it as mandatory content.

An optional shaped reward is:

```text
r_shaped = r_task + beta*(gamma*Phi(s') - Phi(s))
```

`Phi` is a fixed numerical measure of progress and `beta` scales its contribution. The same discount must be used in shaping and learning. For an episodic implementation, use zero potential at terminal states so a residual terminal term does not change the objective. Treat time limits consistently. Under the appropriate MDP/discount/terminal assumptions, this construction preserves optimal policies; arbitrary milestone bonuses need not. See [Ng, Harada, and Russell, Policy Invariance Under Reward Transformations (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

Start with an understandable task reward before adding shaping. Compare shaped and unshaped training while evaluating the original objective. Check loops such as dropping/recollecting items or repeatedly triggering a progress event. Potential shaping is not a guarantee of fast learning or of transfer to real Minecraft.

**Exercise:** For gamma `0.9`, beta `1`, `Phi(s)=2`, and `Phi(s')=3`, compute the added reward. Explain why setting the terminal potential carefully matters.

## 8. Bridge to actual Minecraft

Read `8.pdf` pp. 26–28 on the consequences of an inaccurate model. A custom simulator is a deliberate approximation. A learned policy may exploit its simplified resources, movement, timing, or observations and fail when those assumptions change.

First test whether the real-game interface reliably observes state, executes an action, detects completion/failure, and records outcomes. Then evaluate comparable subtasks. Keep controller details visible: a pathfinding routine, crafting recipe executor, or combat macro is scripted even if a learned policy decides when to invoke it. This is a hybrid controller, and its learned contribution needs a baseline or ablation.

If a macro lasts multiple game ticks, decide what a decision step means. For a tick-based return, aggregate the intervening discounted rewards and use `gamma^k` for a macro lasting `k` ticks. Alternatively, explicitly define a decision-level objective with one discount per macro; that changes the meaning of time. Do not mix the conventions without explanation.

Prepared End fights can isolate combat learning. They do not establish full fresh-world Survival. The final pipeline also has to handle gathering, equipment, survival, Nether resources, stronghold/portal progression, and recovery according to the chosen rules. Maintain separate evidence for each stage and for uninterrupted complete runs.

**Exercise:** Trace one action from the learned policy to actual game behavior. List every scripted operation and every observation it uses.

## 9. Add DQN only when evidence justifies it

Read `7.pdf` pp. 4–35, 36–51, and 57–62; use `course_summary.pdf` pp. 6–8 for revision.

The table becomes impractical when the encoded state space is too large or when nearby states need shared knowledge. DQN replaces the table with a neural network that outputs one Q-value per discrete action. It still uses a Q-learning target and epsilon-greedy exploration, but adds optimization and stability issues.

Before adding it, show a concrete table-size or generalization bottleneck and keep the tabular baseline. Learn a small network, replay buffer, separate target network, gradient updates, and checkpointing first. Structured game observations are a smaller next step than learning directly from pixels. Planned later access to an RTX 5080 is not a reason by itself to introduce a network; current development runs on the laptop.

DQN is conditional extension work, not implemented evidence or a promise that the full Survival goal will become tractable. Do not add PPO, DAgger, inverse RL, general value functions, or distributional RL merely because they appear in the course. `9.pdf`, `14.pdf`, and summary pp. 9–15 provide context for later choices.

## Team understanding and presentation

Each member should be able to answer these questions without reading a generated explanation:

- What state information is available, and which important information is omitted?
- What is one agent action, and which underlying operations are scripted?
- What differs between the Q-learning and SARSA targets?
- Why is terminal handling different from an arbitrary rollout cutoff?
- How could the reward be exploited, and what experiment checks that?
- Which result comes from simulation, which from a prepared game encounter, and which from full Survival?
- How were hyperparameters selected, and what uncertainty remains?
- What did each person implement, verify, analyze, and write?

Practice on a small trace before presenting Minecraft footage. Retain AI assistance and contribution records as described in [course_requirements.md](course_requirements.md), verify generated claims/code, and attribute sources. A convincing presentation explains both achieved behavior and the remaining limitations.
