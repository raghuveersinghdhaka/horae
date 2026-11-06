# HORAE

Simulator and evaluation code for HORAE, a grant-free random access rule in which a device transmits with a
probability set by the hazard h_i = w_i v_i A_i and an entry factor regulated from the idle slots of a broadcast
frame. In its refractory form, HORAE-R, the entry factor is capped at S/(4N) and an active source may attempt only
when its hazard has reached 2N/S, unless its report would change the collector's answer.

## 1. Regenerate the numbers and the figures of the article

Needs Python 3.10 with numpy, scipy, pandas (step 3 only) and matplotlib. From the top folder of this repository:

```
python code/make_numbers.py
python code/make_figures.py
```

The first command writes `generated/numbers.tex`, with every number and table body of the article, from the stored
results in `code/results/`. The second writes the plotted figures to `figures/`. Both take a few minutes.

The rule of the article's headline comparison is the policy `HORAE-R` of `code/run_studies.py`; `calib` is the study
that selected its settings. `HORAE` there is the same rule without the cap, the threshold, the exemption and the
guard's hold and downward factor.

## 2. Run the studies again

```
python code/run_studies.py --list
python code/run_studies.py main fields
```

Each study appends to `code/results/ext_<study>.jsonl` and skips the runs it finds there, so delete a file to repeat
its study. All studies together take several hours on a multi-core machine. `run_experiments.py` and
`extra_experiments.py` produce `code/results/results.json`, the base experiments.

`analysis_checks.py` writes the numerical checks of the propositions, and `load_map.py` the offered load of each
population, `code/results/load_map.json`. `device_loop.c` is one slot of the device
loop with every operation counted; compile it with a C compiler and redirect its output to
`code/results/device_loop.txt`, and with `-DRULE_R` to `code/results/device_loop_r.txt`.

## 3. Rebuild the fields from the public data

`code/data/` holds the prepared fields. `intel_field.npz` and `beijing_field.npz` were prepared from the Intel and
Beijing traces named below (short gaps carried forward, a row with a longer gap dropped) and are used as shipped.
`python code/prepare_fields.py <folder of the downloads>` rebuilds the EPA and PEMS fields from the public files named
in the Data availability section of the article, which it expects as `aqs/hourly_88101_2023.zip`,
`o3/hourly_44201_2023.zip` and `pems/pems_sf.zip` inside that folder:

- United States EPA Air Quality System, hourly PM2.5 and ozone files for 2023,
  https://aqs.epa.gov/aqsweb/airdata/download_files.html
- PEMS-SF, UCI Machine Learning Repository, https://doi.org/10.24432/C52G70

The Intel Berkeley Research Lab trace is at http://db.csail.mit.edu/labdata/labdata.html and the Beijing Multi-Site
Air-Quality data at https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.
