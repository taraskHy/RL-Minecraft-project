# Initial-Q study

**SIMULATOR ONLY; NOT MINECRAFT EVIDENCE**

Unmasked, 50 training episodes, 20 agent seeds per cell, 200 frozen evaluation episodes per agent; simulator search 0.75, chop 0.9, macro 0.95. An untried choice is a frozen decision that picked an action never updated in that state.

| Policy | Initial Q | Mean success | Range over seeds | Seeds below 0.9 | Training successes (of 50) | Mean decisions | Untried choices |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q_learning | 0 | 1.000 | 1.00–1.00 | 0 | 38.8 | 15.7 | 0.0% |
| q_learning | -0.1 | 1.000 | 1.00–1.00 | 0 | 39.2 | 15.9 | 0.0% |
| q_learning | -0.5 | 1.000 | 1.00–1.00 | 0 | 39.2 | 16.1 | 0.0% |
| q_learning | -1 | 1.000 | 1.00–1.00 | 0 | 38.5 | 15.8 | 0.0% |
| q_learning | -1.5 | 0.950 | 0.49–1.00 | 2 | 36.5 | 18.0 | 0.0% |
| q_learning | -2 | 0.650 | 0.00–1.00 | 8 | 30.2 | 31.1 | 0.0% |
| q_learning | -2.5 | 0.250 | 0.00–1.00 | 16 | 12.1 | 48.9 | 0.0% |
| q_learning | -3 | 0.050 | 0.00–0.51 | 20 | 7.6 | 57.7 | 0.0% |
| q_learning | -5 | 0.025 | 0.00–0.51 | 20 | 4.2 | 58.8 | 0.0% |
| sarsa | 0 | 1.000 | 1.00–1.00 | 0 | 37.7 | 15.9 | 0.0% |
| sarsa | -0.1 | 1.000 | 1.00–1.00 | 0 | 37.5 | 15.6 | 0.0% |
| sarsa | -0.5 | 1.000 | 1.00–1.00 | 0 | 37.7 | 16.1 | 0.0% |
| sarsa | -1 | 1.000 | 1.00–1.00 | 0 | 36.2 | 15.5 | 0.0% |
| sarsa | -1.5 | 1.000 | 1.00–1.00 | 0 | 36.0 | 15.7 | 0.0% |
| sarsa | -2 | 1.000 | 1.00–1.00 | 0 | 33.6 | 15.7 | 0.0% |
| sarsa | -2.5 | 0.700 | 0.00–1.00 | 6 | 20.6 | 29.1 | 0.0% |
| sarsa | -3 | 0.325 | 0.00–1.00 | 14 | 13.3 | 45.4 | 0.0% |
| sarsa | -5 | 0.075 | 0.00–1.00 | 19 | 8.0 | 56.6 | 0.0% |
