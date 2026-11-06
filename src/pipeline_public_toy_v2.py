
import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else 'toy_eeg.csv'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(OUT_DIR, exist_ok=True)  # [BUG 2] le dossier de sortie est maintenant garanti d'exister

SF = 250  # Hz, deduit de l'intervalle d'echantillonnage (4ms) dans toy_eeg.csv
WINDOW_S = 1.0
WIN = int(SF * WINDOW_S)

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45),
}

# ordre suppose EEG1..EEG8 -- NON VERIFIE contre une table de montage explicite,
# voir la note en tete de fichier.
CHANNELS = ['O2', 'O1', 'CP2', 'CP1', 'CP6', 'CP5', 'AF8', 'AF7']

NYQ = SF / 2.0
_filter_fail_count = 0


def bandpower_from_psd(f, pxx, band):
    """Puissance spectrale absolue sur la bande, a partir d'une PSD deja calculee."""
    idx = (f >= band[0]) & (f <= band[1])
    if not idx.any():
        return np.nan
    return np.trapezoid(pxx[idx], f[idx])


def spectral_entropy_from_psd(pxx):
    psd_norm = pxx / (pxx.sum() + 1e-12)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))


def bandpass_filter(sig, band):
    """
    [BUG 1 -- fix] Filtre passe-bande zero-phase (filtfilt), pour calculer Energy
    et Peak-to-Peak sur le signal REELLEMENT filtre dans la bande, comme l'exige la
    specification (PPTX, diapo 3). Renvoie None si le filtrage echoue (fenetre trop
    courte, bande degenere) ; l'appelant journalise et met NaN.
    """
    global _filter_fail_count
    lo, hi = band
    try:
        b, a = butter(4, [lo / NYQ, min(hi / NYQ, 0.999)], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        _filter_fail_count += 1
        return None


def extract_window_features(window_df):
    row = {}
    for i, ch in enumerate(CHANNELS, start=1):
        sig = window_df[ch].values

        # [AMELIORATION] une seule PSD par canal, reutilisee pour l'entropie et
        # les 5 bandes (au lieu de 6 appels a welch() sur la v1).
        f, pxx = welch(sig, SF, nperseg=len(sig))
        row[f'EEG{i}_Entropy'] = spectral_entropy_from_psd(pxx)

        for band_name, band_range in BANDS.items():
            row[f'EEG{i}_{band_name}_Power'] = bandpower_from_psd(f, pxx, band_range)

            filtered = bandpass_filter(sig, band_range)
            if filtered is None:
                row[f'EEG{i}_{band_name}_Energy'] = np.nan
                row[f'EEG{i}_{band_name}_Peak_to_Peak'] = np.nan
            else:
                # [BUG 1 -- fix] calcules sur le signal FILTRE, plus sur le brut.
                row[f'EEG{i}_{band_name}_Energy'] = np.sum(filtered ** 2)
                row[f'EEG{i}_{band_name}_Peak_to_Peak'] = filtered.max() - filtered.min()

        alpha_p = row[f'EEG{i}_Alpha_Power']
        beta_p = row[f'EEG{i}_Beta_Power']
        row[f'EEG{i}_Alpha_Beta_Ratio'] = alpha_p / (beta_p + 1e-12)
    return row


df = pd.read_csv(IN_PATH)
print(f"Signal charge : {df.shape}, colonnes = {list(df.columns)}")

rows = []
labels = []
n_windows = len(df) // WIN
n_skipped_mixed = 0
for w in range(n_windows):
    seg = df.iloc[w * WIN:(w + 1) * WIN]
    if seg['soa_condition'].nunique() > 1:
        n_skipped_mixed += 1
        continue  # fenetre a cheval sur 2 conditions -> exclue
    feats = extract_window_features(seg)
    rows.append(feats)
    labels.append(seg['soa_condition'].iloc[0])

features_df = pd.DataFrame(rows)
features_df.insert(0, 'soa_condition', labels)

print(f"Fenetres : {n_windows} totales, {n_skipped_mixed} exclues (a cheval sur 2 conditions), "
      f"{len(features_df)} retenues")
if _filter_fail_count:
    warnings.warn(f"{_filter_fail_count} filtrages passe-bande ont echoue -> NaN en sortie "
                 f"pour ces cellules (fenetre trop courte pour la bande concernee).")

print("Shape obtenue:", features_df.shape)
print("Colonnes (premieres 12):", list(features_df.columns[:12]))
print(features_df.head(3))

# --- Verification de non-regression : les colonnes Energy/PtP ne doivent PLUS
# etre identiques d'une bande a l'autre (c'etait le bug) ---
bands = list(BANDS.keys())
for i in [1]:  # verification rapide sur le premier canal, suffisant en pratique
    e_cols = [f'EEG{i}_{b}_Energy' for b in bands]
    p_cols = [f'EEG{i}_{b}_Peak_to_Peak' for b in bands]
    dup_e = (features_df[e_cols].nunique(axis=1) == 1).mean()
    dup_p = (features_df[p_cols].nunique(axis=1) == 1).mean()
    print(f"\n[controle non-regression] EEG{i} -- lignes ou les 5 bandes Energy sont "
          f"identiques : {dup_e:.1%} (doit etre proche de 0%)")
    print(f"[controle non-regression] EEG{i} -- lignes ou les 5 bandes Peak_to_Peak sont "
          f"identiques : {dup_p:.1%} (doit etre proche de 0%)")

out_path = os.path.join(OUT_DIR, 'toy_eeg_features_extraites.csv')
features_df.to_csv(out_path, index=False)
print(f"\nEcrit : {os.path.abspath(out_path)}")
