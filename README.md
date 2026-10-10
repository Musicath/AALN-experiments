# AALN experiments

Code and numerical data for the experiments in **Activation-Adjacent
Normalization: Approximation Power of RMSNorm and LayerNorm**. The manuscript
source is maintained separately in [Musicath/AALN](https://github.com/Musicath/AALN).
This repository was extracted from that repository at commit `ac3c057`.

The plotted curves evaluate explicit approximation constructions. They do **not**
come from training or optimization. All normalization operations are exact (no
numerical stabilizer), and the target networks have one-dimensional input and
output. The reported maximum error is measured on an 8,193-point grid over
`[-1, 1]`; it is a numerical estimate, not a certified continuous supremum.

## Setup

Use Python 3.11. From the repository root, create and activate a virtual
environment (`.venv\Scripts\Activate.ps1` on PowerShell or
`source .venv/bin/activate` on Linux/macOS), then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The published construction sweeps run on CPU; a GPU is not required.

## Reproduce the paper figures

The committed CSVs under `figures/` are the numerical values used for the
current paper figures. To regenerate the **main-text Figure 3** from those
values:

```bash
python figures/make_representative_m_experiments.py
```

This writes `representative_experiment_{a,b,c}.{pdf,png}` and the shared
legend to `figures/`. Each curve is the mean over six independently generated
width-16 target networks. The three plots correspond to unbounded-range AN
(ReLU), affine PN (tanh), and homogeneous PN (ReLU), respectively.

To recompute the **five-regime activation-family sweep** and its raw data from
scratch, then regenerate Figure 3:

```bash
python experiments/plot_activation_family_sweep.py --output-prefix figures/ln_rms_activation_family_sweep
python figures/make_representative_m_experiments.py
```

The sweep covers four target widths, six targets per width, three activations
per regime, and 11 augmentation values for each normalization. Its full raw
table therefore has 3,960 rows per normalization. The script also writes
tail-slope CSVs and full/compact appendix plots. The default random seed is
recorded in the script; the paper reports the number of targets rather than a
particular seed.

To reproduce the **shifted-ReLU appendix figures**:

```bash
python experiments/plot_shifted_relu_obstruction.py --threshold 10 --output-prefix figures/shifted_relu_10_pn_obstruction
python experiments/plot_shifted_relu_random_teachers.py --threshold 10 --output-prefix figures/shifted_relu_10_random_teachers
```

The first figure compares explicit PN constructions with the finite-width
barrier for a prescribed target. The second repeats the comparison across
random target networks. Both scripts write a plot and the underlying CSV.

## Files and checks

- `experiments/theory_core.py`: exact LN/RMSNorm, activations, and target-network helpers.
- `experiments/plot_*.py`: construction evaluations and appendix plots.
- `figures/make_representative_m_experiments.py`: Figure 3 panels from the committed sweep CSV.
- `figures/*_raw.csv` and `figures/*slope*.csv`: numerical data behind the plots and slope table.
- `tests/`: lightweight checks for normalization and the committed data layout.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

The original manuscript repository also contains an exploratory
construction-versus-training grid. It is not needed to reproduce the figures
reported in this paper, so that training workflow is deliberately excluded
here.

## License

The code and data in this repository are released under the [MIT License](LICENSE).
