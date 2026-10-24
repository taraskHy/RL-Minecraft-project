# Initial-Q study

**SIMULATOR ONLY; NOT MINECRAFT EVIDENCE**

Unmasked, 50 training episodes, 20 agent seeds per cell, 200 frozen evaluation episodes per agent; simulator search 0.35, chop 0.6, macro 0.95. An untried choice is a frozen decision that picked an action never updated in that state.

| Policy | Initial Q | Mean success | Range over seeds | Seeds below 0.9 | Training successes (of 50) | Mean decisions | Untried choices |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_learning | 0 | 1.000 | 1.00–1.00 | 0 | 30.1 | 21.7 | 0.0% |
| q_learning | -0.1 | 1.000 | 1.00–1.00 | 0 | 29.6 | 21.7 | 0.0% |
| q_learning | -0.5 | 1.000 | 1.00–1.00 | 0 | 30.4 | 21.7 | 0.0% |
| q_learning | -1 | 1.000 | 1.00–1.00 | 0 | 30.5 | 21.6 | 0.0% |
| q_learning | -1.5 | 0.723 | 0.00–1.00 | 9 | 21.3 | 32.0 | 0.0% |
| q_learning | -2 | 0.145 | 0.00–1.00 | 18 | 7.9 | 54.3 | 0.0% |
| q_learning | -2.5 | 0.000 | 0.00–0.00 | 20 | 0.1 | 60.0 | 0.0% |
| q_learning | -3 | 0.000 | 0.00–0.00 | 20 | 0.1 | 60.0 | 0.0% |
| q_learning | -5 | 0.000 | 0.00–0.00 | 20 | 0.3 | 60.0 | 0.0% |
| sarsa | 0 | 0.950 | 0.00–1.00 | 1 | 27.4 | 23.4 | 0.0% |
| sarsa | -0.1 | 0.950 | 0.00–1.00 | 1 | 27.6 | 23.5 | 0.0% |
| sarsa | -0.5 | 1.000 | 1.00–1.00 | 0 | 28.7 | 21.3 | 0.0% |
| sarsa | -1 | 1.000 | 1.00–1.00 | 0 | 28.3 | 21.5 | 0.0% |
| sarsa | -1.5 | 0.950 | 0.00–1.00 | 1 | 23.4 | 23.6 | 0.0% |
| sarsa | -2 | 0.576 | 0.00–1.00 | 11 | 16.1 | 37.7 | 0.0% |
| sarsa | -2.5 | 0.050 | 0.00–1.00 | 19 | 1.6 | 58.1 | 0.0% |
| sarsa | -3 | 0.100 | 0.00–1.00 | 18 | 2.1 | 56.0 | 0.0% |
| sarsa | -5 | 0.000 | 0.00–0.00 | 20 | 0.1 | 60.0 | 0.0% |
