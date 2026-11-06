# Methodology note: Level 3 on the lab dataset (v2)

Data: 24 subjects, 5,888 paired EEG+fNIRS trials (5,863 after NaN removal). Everything below was run end to end on the real data.

## 1. Three real bugs found by running the code

**Bug 1: `attention_level` missing from the endo file (guaranteed crash).** Panel (d) of Figure 1 read `endo['attention_level']`, but the endo file has `q` instead; only exo has `attention_level`. The script crashed on that line every run. `q` and `attention_level` carry the same categories (High/Medium/Low) and the same bijective mapping to `cluster` (0 = High, 1 = Low, 2 = Medium), verified before patching. The patch uses whichever column exists and otherwise rebuilds the label from `cluster`.

**Bug 2: corrupted column name in endo.** endo contains `expe+B1:AA1rience_name` (a leftover Excel formula) where exo has `experience_name`. Column-name normalisation added.

**Bug 3: `LPC` and `ConflictSP` are the same column.** They are byte-identical (maximum difference 0.0 over 6,144 rows), not merely correlated at 0.999. Keeping both:

- inflates the number of ERP features (8 → actually 7),
- halves their apparent RF importance, since the forest splits across two copies of one signal,
- would make any table listing LPC and ConflictSP as distinct components wrong.

This is an upstream bug in descriptor extraction and should be fixed at the source. v2 detects and drops exactly duplicated columns automatically.

## 1b. `pipeline_public_toy.py`: two more bugs

This script rebuilds, from a toy raw EEG signal, the same feature family as the lab files (Energy, Power, Peak-to-Peak, Entropy, Alpha/Beta ratio per band and channel), to see what is reusable as is. No `toy_eeg.csv` was provided; the fix was verified on a synthetic signal with the expected structure (8 channels, a `soa_condition` column, 250 Hz).

**Bug 4: Energy and Peak-to-Peak not filtered per band (the important one).** v1 computed both on the raw signal inside the band loop:

```python
row[f'EEG{i}_{band_name}_Energy']       = np.sum(sig**2)          # sig = raw signal
row[f'EEG{i}_{band_name}_Peak_to_Peak'] = sig.max() - sig.min()
```

Neither depends on `band_name`, so the five Energy columns of a channel are identical, as are the five Peak-to-Peak columns (100% of rows on all 8 channels). Two independent confirmations that this is a bug and not a choice:

- the feature specification says Energy is the total energy of the *band-filtered* signal and Peak-to-Peak is the peak-to-peak amplitude of the *filtered* signal;
- the real lab descriptor files do vary by band (0% identical rows), so the production pipeline filters correctly and only this script does not.

Fix: 4th-order Butterworth band-pass with `filtfilt` (zero phase) before computing Energy and Peak-to-Peak. After the fix, on a synthetic signal with theta (~6 Hz) and alpha (~10 Hz) components, Power, Energy and Peak-to-Peak all point to theta then alpha. Cost is +0.1 s on 40 windows × 8 channels, offset by computing one PSD per channel instead of six.

**Bug 5: non-existent output directory (guaranteed crash).** The script wrote to a hard-coded folder that does not exist on a fresh machine. Paths are now arguments and the output directory is created if missing.

**Open point (not a code bug).** `CHANNELS = ['O2','O1','CP2','CP1','CP6','CP5','AF8','AF7']` assumes an `EEG1..EEG8` ↔ site mapping. The spec confirms "8 active sites numbered 1 to 8, spread frontal/central/parietal, referenced to Cz" but gives no number↔site table. Confirm before any spatial interpretation (e.g. frontal asymmetry).

## 2. What changed from v1 to v2

| Point | v1 | v2 |
|---|---|---|
| 3. Hyperparameters | `max_depth=6, min_samples_leaf=5` copied from the public notebook | Nested CV: inner `GridSearchCV` (`GroupKFold` over training subjects) + outer LOSO |
| 4. Multicollinearity | observed only | (a) drop duplicated columns; (b) hierarchical clustering on \|Spearman\| (threshold 0.9) on each fold's training set; (c) standardised Ridge as a comparison |
| 5. Small N | 4-group `GroupKFold`, ±0.24 | LOSO, 24 folds, one score per subject, so paired tests are possible |
| 6. Single target | RT | RT, Accuracy (AUC + `class_weight='balanced'`), Congruence decoding, Mode decoding |
| 7. Convergence | absent | Spearman between RF importance and paired univariate Cohen's d |
| 8. Multiple comparisons | 5+ uncorrected tests | one pre-designated confirmatory comparison + Benjamini–Hochberg-corrected exploratory block |

Added beyond the list because they change how results read:

- a **train-mean baseline**, without which a negative LOSO R² is uninterpretable;
- a **paired test on per-subject MAE** in addition to R², since per-subject R² depends on the subject's own variance and is unstable (see §4).

## 3. Results (fast profile, 24 LOSO folds)

### Confirmatory (pre-designated, uncorrected)

| Model | R² (LOSO) | median |
|---|---|---|
| Fusion EEG+fNIRS+SOA | −0.180 ± 0.389 | −0.089 |
| SOA-only floor | −0.205 ± 0.524 | −0.055 |

Paired Wilcoxon (24 subjects): R² p = 0.79; MAE p = 0.46.

The initial −0.13 is not a tuning problem: the inner CV selected `max_depth=6, min_samples_leaf=20` unanimously over 24 folds (a stable setting, different from the `min_samples_leaf=5` copied from the public notebook) and the score barely moves. There is no signal beyond the design floor.

### Exploratory (FDR-corrected)

| Comparison | Δ mean | p | p FDR | Significant |
|---|---|---|---|---|
| RT: EEG+SOA vs floor (MAE) | +0.79 ms | 0.456 | 0.628 | no |
| RT: fNIRS+SOA vs floor (MAE) | +0.08 ms | 0.061 | 0.182 | no |
| RT: fusion vs floor (Ridge, MAE) | +8.47 ms | 0.966 | 0.966 | no |
| Accuracy: fusion vs floor (AUC) | +0.030 | 0.128 | 0.256 | no |
| Congruence: AUC vs chance | +0.038 | 0.0005 | 0.0030 | yes |
| Mode: AUC vs chance | +0.025 | 0.524 | 0.628 | no |

The only positive result, and it survives FDR correction: congruence is decodable above chance from the signal (AUC = 0.538 ± 0.040, median 0.539). A small but remarkably consistent effect across subjects.

**Argument structure this supports.** The EEG/fNIRS signal contains information about trial type (congruence) but does not predict RT beyond the design floor. That separates *information present in the signal* from *information usable to predict behaviour trial by trial at N = 24*, which is more interesting than a plain null.

### Convergence

Spearman(\|Cohen's d\|, fusion RF importance): ρ = +0.714, p = 0.071 (7 components). Right direction, N too small for significance, so present it as qualitative convergence, not a test. Against the congruence model's importances, ρ = 0.000: importances there are nearly uniform (~0.16 each) and discriminate nothing. Say so rather than omit it.

## 4. Points to state explicitly in any write-up

- **P3a has importance 0.0000. That means "removed as redundant", not "uninformative".** P3a correlates at ρ = 0.945 with N2 and was eliminated by the clustering in all 24 folds. A reviewer seeing an exact zero will ask.
- **fNIRS adds nothing.** `mean_hbo` and `mean_hbr` have importances of 0.0000 and 0.0002 in the fusion model. Averaging over 12 channels probably destroys spatial specificity. Frame it as a feature-engineering limit, not proof that fNIRS is useless.
- **Per-subject R² is misleading at this N.** Subject 18 has R² = −1.82, yet its MAE (96.8 ms) beats the mean baseline (109.1 ms): the R² is poor because that subject's own RT variance is low, not because the model is bad. Overall MAE: fusion 85.7 ms vs train-mean baseline 86.8 ms. This is why the confirmatory test is reported on MAE as well as R².
- **Remaining limitation.** The fusion Ridge stays unstable (R² = −0.563 ± 1.344): one or two subjects blow up the mean while the median delta is ≈ 0. Report the median or switch to a Huber loss.
- **Mode-decoding caveat.** Endo and exo are probably recorded in separate blocks, so above-chance Mode decoding could reflect session drift (sensor drift, impedance) rather than cognition. The AUC is not significant here, so it does not arise, but if a `full`-profile run makes it significant, a temporal-block control is required.

## 5. How to re-run

```bash
python src/private_dataset/analyse_descripteurs_v2.py path/to/dataset/
python src/private_dataset/adapter_prive_level3_v2.py path/to/dataset/ full
```

- **fast** (default): reduced grid, 150-tree forests, ~20 min on one core. Produced the numbers above; methodologically identical, just a coarser grid.
- **full**: complete grid (`max_depth` ∈ {4, 6, 8, None} × `min_samples_leaf` ∈ {1, 5, 20}), 500 trees, 5 inner folds, `n_jobs=-1`. Run before submission and report this profile. Since inner CV chose the same setting in 24/24 folds, results should not move.
- **smoke**: end-to-end check in seconds (6 subjects); results not interpretable.

Outputs: `resultat_confirmatoire`, `resultats_exploratoires_FDR`, `importance_fusion`, `scores_par_sujet_fusion`, `convergence_rf_vs_univarie`, `effets_univaries_erp`, `fig1_*.png`, `fig2_*.png`.
