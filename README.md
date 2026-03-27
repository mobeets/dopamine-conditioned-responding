# "Phasic dopamine drives conditioned responding beyond its role in learning"

## Setup

- Install [uv](https://docs.astral.sh/uv/#installation)
- Run `uv sync`
- Download experimental data:
    - [Matsumoto et al. (2016)]('https://doi.org/10.5061/dryad.8cz8w9gxj')
    - Eshel et al. (2016)
    - Kim et al. (2020)
    - [Amo et al. (2024)]('https://datadryad.org/stash/dataset/doi:10.5061/dryad.ncjsxkt25')
    - Qian & Burrell et al. (2025)

## Run

- Figures 1, 2, 4: `uv run 1_2_4_dopamine_licking.py`
- Figures 3, 5C: `uv run 3_5_other_experiments.py`
- Figure 5B, 5D: `uv run 5_td_phenotypes.py`
- Figure 6: `uv run 6_spike_triggered_licks.py`
- Figure 7: `uv run 7_causal_dopamine_perts.py`
