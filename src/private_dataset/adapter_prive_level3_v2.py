"""
Adaptateur v2 : methodologie Level 3 (public) appliquee au dataset prive fusionne.
==================================================================================
Ce qui est repris du notebook public : logique de validation par sujet, comparaison
EEG-seul / fNIRS-seul / fusion / modele-plancher base sur le design experimental.

Corrections methodologiques par rapport a la v1 (points 3 a 8) :
  [3] Hyperparametres : validation croisee IMBRIQUEE. Boucle interne GroupKFold
      sur les sujets d'entrainement pour choisir max_depth / min_samples_leaf ;
      boucle externe LOSO pour l'evaluation. Aucun hyperparametre n'est plus
      copie du notebook public sans verification.
  [4] Multicolinearite TRAITEE, pas seulement constatee :
      (a) suppression des colonnes exactement dupliquees (LPC == ConflictSP) ;
      (b) clustering hierarchique sur la correlation de Spearman, seuil 0.9,
          calcule sur le TRAIN de chaque fold uniquement (pas de fuite) ;
      (c) modele regularise Ridge (StandardScaler + RidgeCV dans un pipeline
          ajuste fold par fold) en comparaison de la foret aleatoire.
  [5] LOSO (24 folds) au lieu du GroupKFold a 4 groupes : intervalle de
      confiance interpretable au lieu d'un ecart-type sur 4 valeurs.
  [6] Quatre cibles au lieu d'une : RT, Accuracy (desequilibre 93/7 gere par
      class_weight + AUC au lieu du taux de bonne classification), decodage de
      Congruence et de Mode.
  [7] Convergence : correlation de Spearman entre l'importance RF des
      composantes ERP et leur taille d'effet univariee (Cohen's d apparie par
      Congruence, produite par analyse_descripteurs_v2.py).
  [8] Comparaisons multiples : UNE comparaison confirmatoire pre-designee
      (fusion vs plancher SOA sur RT). Tout le reste est etiquete exploratoire
      et corrige par Benjamini-Hochberg (FDR).

Ajouts au-dela de la liste :
  - Baseline "moyenne du train" explicite, sans laquelle un R2 negatif en LOSO
    n'est pas interpretable (un R2 de -0.2 peut etre normal si la variance
    inter-sujets domine).
  - Test apparie sur l'erreur absolue par sujet en plus du R2 par sujet : le R2
    par sujet depend de la variance propre du sujet et est instable.
  - Journal des essais/sujets exclus par les NaN.

Usage :
    python3 adapter_prive_level3_v2.py [chemin/dataset/] [fast|full]
        fast : grille reduite, foret plus petite (verification, mono-coeur)
        full : grille complete, n_jobs=-1 (a lancer sur une vraie machine)
"""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

BASE = sys.argv[1] if len(sys.argv) > 1 else './dataset_nos_donnes/'
if not BASE.endswith(os.sep):
    BASE += os.sep
PROFILE = sys.argv[2] if len(sys.argv) > 2 else 'fast'
OUT = os.environ.get('OUTDIR', '.')
os.makedirs(OUT, exist_ok=True)

if PROFILE == 'full':
    CFG = dict(n_jobs=-1, trees_inner=200, trees_outer=500, inner_folds=5,
               grid={'max_depth': [4, 6, 8, None], 'min_samples_leaf': [1, 5, 20]})
elif PROFILE == 'smoke':
    # Test de bout en bout uniquement : resultats NON interpretables.
    CFG = dict(n_jobs=1, trees_inner=8, trees_outer=12, inner_folds=2,
               grid={'max_depth': [4], 'min_samples_leaf': [20]})
else:
    CFG = dict(n_jobs=1, trees_inner=60, trees_outer=150, inner_folds=3,
               grid={'max_depth': [6, None], 'min_samples_leaf': [5, 20]})

SEED = 42
CORR_THRESH = 0.9
print(f"Profil = {PROFILE} | {CFG}\n")


# ==========================================================================
# 1. Chargement, fusion, hygiene des features
# ==========================================================================
fnirs = pd.read_csv(BASE + 'Descripteurs_fNIRS.csv')
glob = pd.read_csv(BASE + 'Descripteurs_EEG_Globaux.csv', sep=';')

KEYS = ['Sujet_ID', 'Mode', 'Congruence', 'SOA', 'Trial']
df = fnirs.merge(glob, on=KEYS, how='inner', suffixes=('_fnirs', '_glob')).copy()

hbo_cols = [c for c in df.columns if c.startswith('HbO_') and c.endswith('_Amp_PtP')]
hbr_cols = [c for c in df.columns if c.startswith('HbR_') and c.endswith('_Amp_PtP')]
df = pd.concat([df,
                df[hbo_cols].mean(axis=1).rename('mean_hbo'),
                df[hbr_cols].mean(axis=1).rename('mean_hbr')], axis=1)

EEG_FEATS = ['N2', 'N450', 'P3', 'P3a', 'LPC', 'MMN', 'ConflictSP', 'CNV']
NIR_FEATS = ['mean_hbo', 'mean_hbr']
SOA_FEAT = ['SOA']
GROUP = 'Sujet_ID'

# [4a] Colonnes exactement dupliquees -> une seule survit.
# Sur ces donnees : LPC et ConflictSP sont byte-identiques. Les garder toutes
# les deux fausse toute lecture de l'importance des variables.
dupes = []
kept = []
for c in EEG_FEATS:
    dup_of = next((k for k in kept if df[c].equals(df[k])), None)
    if dup_of:
        dupes.append((c, dup_of))
    else:
        kept.append(c)
if dupes:
    print("[4a] Colonnes EEG exactement dupliquees, retirees :")
    for c, k in dupes:
        print(f"     {c} est identique a {k} -> {c} retiree")
EEG_FEATS = kept

ALL_FEATS = EEG_FEATS + NIR_FEATS + SOA_FEAT
n_before = len(df)
sub_before = df[GROUP].nunique()
df = df.dropna(subset=ALL_FEATS + ['RT_glob', 'Accuracy_glob'])
print(f"\nEssais : {n_before} -> {len(df)} apres suppression des NaN "
      f"({n_before - len(df)} retires) | sujets : {sub_before} -> {df[GROUP].nunique()}")
print(f"Features EEG retenues ({len(EEG_FEATS)}) : {EEG_FEATS}")

if PROFILE == 'smoke':
    keep = sorted(df[GROUP].unique())[:6]
    df = df[df[GROUP].isin(keep)].copy()
    print(f"[SMOKE] sous-echantillon : {len(df)} essais, {df[GROUP].nunique()} sujets")

groups = df[GROUP].values
n_subjects = df[GROUP].nunique()


# ==========================================================================
# 2. Outils : reduction de colinearite (train only) + CV imbriquee LOSO
# ==========================================================================
def select_uncorrelated(train_X, feat_cols, thresh=CORR_THRESH):
    """
    [4b] Clustering hierarchique sur |Spearman| calcule UNIQUEMENT sur le train
    du fold. Un representant par cluster (celui qui resume le mieux son
    cluster = correlation moyenne intra-cluster maximale). Non supervise :
    la cible n'intervient pas, donc aucune fuite possible.
    """
    if len(feat_cols) < 2:
        return list(feat_cols)
    corr = np.asarray(pd.DataFrame(train_X, columns=feat_cols).corr(method='spearman')).copy()
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1.0 - np.abs(corr), 0, None)
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method='average')
    labels = fcluster(link, t=1.0 - thresh, criterion='distance')
    selected = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) == 1:
            selected.append(feat_cols[idx[0]])
        else:
            sub = np.abs(corr[np.ix_(idx, idx)])
            selected.append(feat_cols[idx[int(np.argmax(sub.mean(axis=1)))]])
    return sorted(selected, key=feat_cols.index)


def nested_loso(data, feat_cols, target, task='reg', label='', reduce_collinearity=True):
    """
    [3][5] Boucle externe LOSO ; boucle interne GroupKFold sur les sujets
    d'entrainement pour selectionner les hyperparametres. Renvoie un score par
    sujet pour permettre les tests apparies.
    """
    t0 = time.time()
    X_all = data[feat_cols]
    y = data[target].values
    g = data[GROUP].values
    logo = LeaveOneGroupOut()

    per_sub, imps = [], []
    for tr, te in logo.split(X_all, y, g):
        sub_id = g[te][0]
        Xtr_df, Xte_df = X_all.iloc[tr], X_all.iloc[te]
        ytr, yte = y[tr], y[te]

        feats = (select_uncorrelated(Xtr_df.values, list(feat_cols))
                 if reduce_collinearity else list(feat_cols))
        Xtr, Xte = Xtr_df[feats].values, Xte_df[feats].values

        n_inner = min(CFG['inner_folds'], len(np.unique(g[tr])))
        inner = GroupKFold(n_splits=max(2, n_inner))

        if task == 'reg':
            base = RandomForestRegressor(n_estimators=CFG['trees_inner'],
                                         random_state=SEED, n_jobs=CFG['n_jobs'])
            scoring = 'neg_mean_absolute_error'
        else:
            base = RandomForestClassifier(n_estimators=CFG['trees_inner'],
                                          class_weight='balanced',
                                          random_state=SEED, n_jobs=CFG['n_jobs'])
            scoring = 'roc_auc'

        gs = GridSearchCV(base, CFG['grid'], cv=inner, scoring=scoring,
                          n_jobs=1, refit=False)
        gs.fit(Xtr, ytr, groups=g[tr])
        best = gs.best_params_

        if task == 'reg':
            mdl = RandomForestRegressor(n_estimators=CFG['trees_outer'], random_state=SEED,
                                        n_jobs=CFG['n_jobs'], **best)
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            rec = {'sujet': sub_id, 'r2': r2_score(yte, pred),
                   'mae': mean_absolute_error(yte, pred),
                   'mae_baseline_moyenne': mean_absolute_error(yte, np.full_like(yte, ytr.mean())),
                   'n_test': len(yte), 'best': json.dumps(best)}
        else:
            mdl = RandomForestClassifier(n_estimators=CFG['trees_outer'], class_weight='balanced',
                                         random_state=SEED, n_jobs=CFG['n_jobs'], **best)
            mdl.fit(Xtr, ytr)
            if len(np.unique(yte)) < 2:          # sujet a classe unique -> AUC indefinie
                rec = {'sujet': sub_id, 'auc': np.nan, 'n_test': len(yte),
                       'best': json.dumps(best)}
            else:
                proba = mdl.predict_proba(Xte)[:, 1]
                rec = {'sujet': sub_id, 'auc': roc_auc_score(yte, proba),
                       'n_test': len(yte), 'best': json.dumps(best)}
        per_sub.append(rec)
        imps.append(pd.Series(mdl.feature_importances_, index=feats))

    res = pd.DataFrame(per_sub)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    metric = 'r2' if task == 'reg' else 'auc'
    vals = res[metric].dropna()
    print(f"  {label:38s} {metric.upper()}(LOSO) = {vals.mean():+.3f} +/- {vals.std():.3f} "
          f"[mediane {vals.median():+.3f}, n={len(vals)}]  ({time.time()-t0:.0f}s)")
    return res, imp


def ridge_loso(data, feat_cols, target, label=''):
    """[4c] Ridge standardise, ajuste fold par fold (le scaler ne voit que le train)."""
    X, y, g = data[feat_cols].values, data[target].values, data[GROUP].values
    out = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        pipe = Pipeline([('sc', StandardScaler()),
                         ('rg', RidgeCV(alphas=np.logspace(-3, 4, 20)))])
        pipe.fit(X[tr], y[tr])
        out.append({'sujet': g[te][0], 'r2': r2_score(y[te], pipe.predict(X[te])),
                    'mae': mean_absolute_error(y[te], pipe.predict(X[te]))})
    r = pd.DataFrame(out)
    print(f"  {label:38s} R2(LOSO) = {r['r2'].mean():+.3f} +/- {r['r2'].std():.3f}  [Ridge]")
    return r


def paired_test(a, b, col, name):
    """Wilcoxon apparie par sujet sur la metrique 'col'."""
    m = pd.merge(a[['sujet', col]], b[['sujet', col]], on='sujet', suffixes=('_A', '_B')).dropna()
    if len(m) < 5:
        return {'comparaison': name, 'n': len(m), 'delta_moyen': np.nan, 'p': np.nan}
    d = m[f'{col}_A'] - m[f'{col}_B']
    stat, p = wilcoxon(m[f'{col}_A'], m[f'{col}_B'])
    return {'comparaison': name, 'n': len(m), 'delta_moyen': d.mean(),
            'delta_median': d.median(), 'p': p}


# ==========================================================================
# 3. CONFIRMATOIRE : une seule comparaison pre-designee [8]
# ==========================================================================
print("\n" + "=" * 78)
print("BLOC CONFIRMATOIRE (pre-designe, non corrige) : RT -- fusion vs plancher SOA")
print("=" * 78)

fused = EEG_FEATS + NIR_FEATS + SOA_FEAT
res_fused, imp_fused = nested_loso(df, fused, 'RT_glob', 'reg', 'Fusion EEG+fNIRS+SOA (primaire)')
res_floor, _ = nested_loso(df, SOA_FEAT, 'RT_glob', 'reg', 'SOA seul (plancher, primaire)',
                           reduce_collinearity=False)

primary_r2 = paired_test(res_fused, res_floor, 'r2', 'RT : fusion vs plancher (R2 par sujet)')
primary_mae = paired_test(res_fused, res_floor, 'mae', 'RT : fusion vs plancher (MAE par sujet)')
print(f"\n  Test confirmatoire (Wilcoxon apparie, {n_subjects} sujets) :")
print(f"    R2  : delta moyen = {primary_r2['delta_moyen']:+.4f}, p = {primary_r2['p']:.4f}")
print(f"    MAE : delta moyen = {primary_mae['delta_moyen']:+.4f} ms, p = {primary_mae['p']:.4f} "
      f"(negatif = la fusion fait mieux)")

# Reference d'interpretation : la baseline "moyenne du train"
print(f"\n  Reference : MAE moyenne du modele fusion        = {res_fused['mae'].mean():.2f} ms")
print(f"              MAE de la baseline 'moyenne du train' = "
      f"{res_fused['mae_baseline_moyenne'].mean():.2f} ms")
print("  -> si les deux sont proches, le R2 negatif traduit la variance inter-sujets,")
print("     pas un modele aberrant.")


# ==========================================================================
# 4. EXPLORATOIRE : tout le reste, corrige FDR [6][8]
# ==========================================================================
print("\n" + "=" * 78)
print("BLOC EXPLORATOIRE (non pre-designe -> correction Benjamini-Hochberg)")
print("=" * 78)

explo = []

print("\n[RT] modalites separees")
res_eeg, imp_eeg = nested_loso(df, EEG_FEATS + SOA_FEAT, 'RT_glob', 'reg', 'EEG (ERP) + SOA')
res_nir, _ = nested_loso(df, NIR_FEATS + SOA_FEAT, 'RT_glob', 'reg', 'fNIRS + SOA')
explo.append(paired_test(res_eeg, res_floor, 'mae', 'RT : EEG+SOA vs plancher (MAE)'))
explo.append(paired_test(res_nir, res_floor, 'mae', 'RT : fNIRS+SOA vs plancher (MAE)'))

print("\n[RT] modeles regularises (colinearite)")
ridge_fused = ridge_loso(df, fused, 'RT_glob', 'Fusion (Ridge standardise)')
ridge_floor = ridge_loso(df, SOA_FEAT, 'RT_glob', 'SOA seul (Ridge standardise)')
explo.append(paired_test(ridge_fused, ridge_floor, 'mae', 'RT : fusion vs plancher (Ridge, MAE)'))

print("\n[Accuracy] desequilibre 93/7 -> AUC + class_weight='balanced'")
df['acc_bin'] = df['Accuracy_glob'].astype(int)
res_acc, imp_acc = nested_loso(df, fused, 'acc_bin', 'clf', 'Accuracy : fusion EEG+fNIRS+SOA')
res_acc_floor, _ = nested_loso(df, SOA_FEAT, 'acc_bin', 'clf', 'Accuracy : plancher SOA',
                               reduce_collinearity=False)
explo.append(paired_test(res_acc, res_acc_floor, 'auc', 'Accuracy : fusion vs plancher (AUC)'))

print("\n[Decodage] Congruence et Mode a partir du signal")
df['congr_bin'] = (df['Congruence'] == 'incompatible').astype(int)
df['mode_bin'] = (df['Mode'] == 'exo').astype(int)
sig_feats = EEG_FEATS + NIR_FEATS      # sans SOA : on decode le signal, pas le design
res_cong, imp_cong = nested_loso(df, sig_feats, 'congr_bin', 'clf', 'Congruence depuis EEG+fNIRS')
res_mode, imp_mode = nested_loso(df, sig_feats, 'mode_bin', 'clf', 'Mode depuis EEG+fNIRS')

for res, nm in [(res_cong, 'Congruence'), (res_mode, 'Mode')]:
    v = res['auc'].dropna()
    stat, p = wilcoxon(v - 0.5) if len(v) >= 5 else (np.nan, np.nan)
    explo.append({'comparaison': f'{nm} : AUC vs hasard (0.5)', 'n': len(v),
                  'delta_moyen': v.mean() - 0.5, 'delta_median': v.median() - 0.5, 'p': p})

explo_df = pd.DataFrame(explo)
ok = explo_df['p'].notna()
explo_df.loc[ok, 'p_FDR'] = multipletests(explo_df.loc[ok, 'p'], method='fdr_bh')[1]
explo_df['significatif_FDR_5pct'] = explo_df['p_FDR'] < 0.05

print("\n--- Resultats exploratoires, p corrigees (Benjamini-Hochberg) ---")
print(explo_df.round(4).to_string(index=False))


# ==========================================================================
# 5. CONVERGENCE : importance RF vs taille d'effet univariee [7]
# ==========================================================================
print("\n" + "=" * 78)
print("CONVERGENCE : importance RF (multivarie) vs Cohen's d (univarie)")
print("=" * 78)

eff_path = next((p for p in [os.path.join(OUT, 'effets_univaries_erp.csv'),
                             'effets_univaries_erp.csv',
                             os.path.join('out', 'effets_univaries_erp.csv')]
                 if os.path.exists(p)), os.path.join(OUT, 'effets_univaries_erp.csv'))
if os.path.exists(eff_path):
    eff = pd.read_csv(eff_path)
    conv = eff[eff['composante'].isin(EEG_FEATS)].copy()
    conv['importance_RF_fusion'] = conv['composante'].map(imp_fused).fillna(0.0)
    conv['importance_RF_congruence'] = conv['composante'].map(imp_cong).fillna(0.0)
    print(conv[['composante', 'cohens_d_apparie', 'abs_d',
                'importance_RF_fusion', 'importance_RF_congruence']].round(4).to_string(index=False))
    for col in ['importance_RF_fusion', 'importance_RF_congruence']:
        sub = conv.dropna(subset=['abs_d', col])
        if len(sub) >= 4:
            rho, p = spearmanr(sub['abs_d'], sub[col])
            print(f"  Spearman(|Cohen's d| , {col}) : rho = {rho:+.3f}, p = {p:.4f}, "
                  f"n = {len(sub)} composantes")
    conv.to_csv(os.path.join(OUT, 'convergence_rf_vs_univarie.csv'), index=False)
else:
    print(f"  [info] {eff_path} absent : lancer d'abord analyse_descripteurs_v2.py")

print("\n--- Importance moyenne des variables (fusion, moyennee sur les 24 folds) ---")
print(imp_fused.round(4).to_string())

# ==========================================================================
# 6. Sauvegardes
# ==========================================================================
pd.DataFrame([primary_r2, primary_mae]).to_csv(os.path.join(OUT, 'resultat_confirmatoire.csv'),
                                               index=False)
explo_df.to_csv(os.path.join(OUT, 'resultats_exploratoires_FDR.csv'), index=False)
imp_fused.rename('importance').to_csv(os.path.join(OUT, 'importance_fusion.csv'))
res_fused.to_csv(os.path.join(OUT, 'scores_par_sujet_fusion.csv'), index=False)
print(f"\nResultats ecrits dans {os.path.abspath(OUT)}")
