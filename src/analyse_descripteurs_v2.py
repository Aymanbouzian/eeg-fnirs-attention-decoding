
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else './dataset_nos_donnes/'
if not BASE.endswith(os.sep):
    BASE += os.sep
OUT = os.environ.get('OUTDIR', '.')
os.makedirs(OUT, exist_ok=True)

ERP_COLS = ['N2', 'N450', 'P3', 'P3a', 'LPC', 'MMN', 'ConflictSP', 'CNV']
ATTENTION_CANDIDATES = ['attention_level', 'q', 'attention', 'niveau_attention']
CLUSTER_TO_LABEL = {0: 'High', 1: 'Low', 2: 'Medium'}  # verifie sur endo ET exo


# --------------------------------------------------------------------------
# 1. Chargement
# --------------------------------------------------------------------------
def normalize_columns(df):
    """Nettoie les noms de colonnes corrompus a l'export Excel."""
    fixed = {}
    for c in df.columns:
        c2 = str(c)
        if 'rience_name' in c2:          # 'expe+B1:AA1rience_name' -> experience_name
            c2 = 'experience_name'
        fixed[c] = c2.strip()
    return df.rename(columns=fixed)


fnirs = pd.read_csv(BASE + 'Descripteurs_fNIRS.csv')
glob = pd.read_csv(BASE + 'Descripteurs_EEG_Globaux.csv', sep=';')  # separateur ; important
endo = normalize_columns(pd.read_excel(BASE + 'descripteurs_EEG_endo_labeled_with_attention2.xlsx'))
exo = normalize_columns(pd.read_excel(BASE + 'descripteurs_EEG_exo_labeled_with_attention2.xlsx'))

print(f"fnirs {fnirs.shape} | glob {glob.shape} | endo {endo.shape} | exo {exo.shape}")


def get_attention_series(df, name):
    """
    [BUG 1] Retourne la serie de niveau d'attention quel que soit son nom de
    colonne. Si aucune colonne texte n'existe, on la reconstruit depuis
    'cluster' (mapping verifie identique sur endo et exo).
    """
    for cand in ATTENTION_CANDIDATES:
        if cand in df.columns:
            if cand != 'attention_level':
                print(f"  [info] {name}: pas de colonne 'attention_level', "
                      f"utilisation de '{cand}' a la place.")
            return df[cand].astype(str), cand
    if 'cluster' in df.columns:
        print(f"  [info] {name}: aucune colonne texte, reconstruction depuis 'cluster'.")
        return df['cluster'].map(CLUSTER_TO_LABEL).astype(str), 'cluster (mappe)'
    raise KeyError(f"{name}: aucune colonne de niveau d'attention exploitable "
                   f"(cherche: {ATTENTION_CANDIDATES + ['cluster']})")


print("\n--- Detection de la colonne de niveau d'attention ---")
endo_att, endo_src = get_attention_series(endo, 'endo')
exo_att, exo_src = get_attention_series(exo, 'exo')

# Controle de coherence label <-> cluster (doit etre une bijection)
for name, df, att in [('endo', endo, endo_att), ('exo', exo, exo_att)]:
    if 'cluster' in df.columns:
        ct = pd.crosstab(att, df['cluster'])
        bijective = (ct > 0).sum(axis=1).max() == 1 and (ct > 0).sum(axis=0).max() == 1
        print(f"  {name}: source='{ '{}'.format(endo_src if name=='endo' else exo_src) }', "
              f"labels={sorted(att.unique())}, bijection label/cluster: {bijective}")
        if not bijective:
            print(f"  [ATTENTION] {name}: le label et le cluster ne coincident pas.\n{ct}")


# --------------------------------------------------------------------------
# 2. Verification de coherence : alignement reel des essais fnirs <-> glob
# --------------------------------------------------------------------------
KEYS = ['Sujet_ID', 'Mode', 'Congruence', 'SOA', 'Trial']
print("\n--- Coherence fnirs <-> glob (alignement essai par essai) ---")

dup_f = fnirs.duplicated(KEYS).sum()
dup_g = glob.duplicated(KEYS).sum()
print(f"Cles dupliquees : fnirs={dup_f}, glob={dup_g} "
      f"(doit etre 0, sinon la fusion cree des lignes fantomes)")

paired = fnirs[KEYS + ['RT', 'Accuracy']].merge(
    glob[KEYS + ['RT', 'Accuracy']], on=KEYS, how='inner', suffixes=('_fnirs', '_glob'))
print(f"Essais apparies : {len(paired)} / fnirs={len(fnirs)} / glob={len(glob)}")

rt_close = np.isclose(paired['RT_fnirs'], paired['RT_glob'], rtol=0, atol=1e-6)
acc_eq = paired['Accuracy_fnirs'] == paired['Accuracy_glob']
print(f"Accord RT (essai par essai)       : {rt_close.mean():.4%}")
print(f"Accord Accuracy (essai par essai) : {acc_eq.mean():.4%}")
if rt_close.mean() < 1.0:
    bad = paired.loc[~rt_close, KEYS + ['RT_fnirs', 'RT_glob']].head(5)
    print("[ATTENTION] desaccord sur RT -- exemples :\n", bad.to_string(index=False))
    print("-> tant que ce n'est pas resolu, la fusion EEG+fNIRS n'est pas fiable.")
else:
    print("-> alignement parfait : la fusion du script d'adaptation est valide.")

# Les moyennes par Mode (comparaison v1) restent utiles comme resume descriptif
print("\nRT moyen par Mode :")
print(pd.concat([fnirs.groupby('Mode')['RT'].mean().rename('fnirs'),
                 glob.groupby('Mode')['RT'].mean().rename('glob')], axis=1).round(2))
print("\nAccuracy moyenne par Mode :")
print(pd.concat([fnirs.groupby('Mode')['Accuracy'].mean().rename('fnirs'),
                 glob.groupby('Mode')['Accuracy'].mean().rename('glob')], axis=1).round(4))


# --------------------------------------------------------------------------
# 3. Figure 1 : comportemental
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) RT par Mode x Congruence, avec barres d'erreur inter-sujets (SEM)
sub_means = glob.groupby(['Sujet_ID', 'Mode', 'Congruence'])['RT'].mean().reset_index()
m = sub_means.groupby(['Mode', 'Congruence'])['RT'].mean().unstack()
s = (sub_means.groupby(['Mode', 'Congruence'])['RT'].std()
     / np.sqrt(sub_means['Sujet_ID'].nunique())).unstack()
m.plot(kind='bar', ax=axes[0, 0], yerr=s, capsize=4, color=['#4C72B0', '#DD8452'])
axes[0, 0].set_title("RT moyen par Mode x Congruence (barres = SEM inter-sujets)")
axes[0, 0].set_ylabel("RT (ms)")
axes[0, 0].tick_params(axis='x', rotation=0)

# (b) RT par SOA, avec SEM inter-sujets
soa_sub = glob.groupby(['Sujet_ID', 'SOA'])['RT'].mean().reset_index()
soa_m = soa_sub.groupby('SOA')['RT'].mean().sort_index()
soa_s = (soa_sub.groupby('SOA')['RT'].std() / np.sqrt(soa_sub['Sujet_ID'].nunique())).sort_index()
axes[0, 1].errorbar(soa_m.index.astype(str), soa_m.values, yerr=soa_s.values,
                    marker='o', capsize=4, color='#55A868')
axes[0, 1].set_title("RT moyen selon le SOA")
axes[0, 1].set_xlabel("SOA (ms)")
axes[0, 1].set_ylabel("RT (ms)")

# (c) Accuracy par Mode x Congruence
acc = glob.groupby(['Mode', 'Congruence'])['Accuracy'].mean().unstack()
acc.plot(kind='bar', ax=axes[1, 0], color=['#4C72B0', '#DD8452'])
axes[1, 0].set_title("Accuracy par Mode x Congruence")
axes[1, 0].set_ylim(0.8, 1.0)
axes[1, 0].tick_params(axis='x', rotation=0)

# (d) [BUG 1 corrige] Repartition du niveau d'attention, en PROPORTIONS
labels = ['High', 'Medium', 'Low']
x = np.arange(len(labels))
width = 0.35
endo_p = endo_att.value_counts(normalize=True)
exo_p = exo_att.value_counts(normalize=True)
axes[1, 1].bar(x - width / 2, [endo_p.get(l, 0) for l in labels], width,
               label=f"Endo (n={len(endo_att)}, src={endo_src})")
axes[1, 1].bar(x + width / 2, [exo_p.get(l, 0) for l in labels], width,
               label=f"Exo (n={len(exo_att)}, src={exo_src})")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_ylabel("Proportion d'essais")
axes[1, 1].set_title("Repartition du niveau d'attention")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_comportemental.png'), dpi=130)
plt.close(fig)


# --------------------------------------------------------------------------
# 4. Figure 2 : signal neuronal
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

key_erp = ['N2', 'P3', 'MMN', 'CNV']
glob.groupby('Congruence')[key_erp].mean().T.plot(kind='bar', ax=axes[0])
axes[0].set_title("Amplitude ERP moyenne par Congruence")
axes[0].tick_params(axis='x', rotation=0)

chans = [f"HbO_Ch{i}_Amp_PtP" for i in range(1, 13)]
means_by_mode = fnirs.groupby('Mode')[chans].mean()
for mode in means_by_mode.index:
    axes[1].plot(range(1, 13), means_by_mode.loc[mode].values, marker='o', label=mode)
axes[1].set_title("Amplitude HbO (Peak-to-Peak) par canal")
axes[1].set_xlabel("Canal fNIRS")
axes[1].legend(title="Mode")

corr = glob[ERP_COLS].corr()
im = axes[2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[2].set_xticks(range(len(ERP_COLS)))
axes[2].set_yticks(range(len(ERP_COLS)))
axes[2].set_xticklabels(ERP_COLS, rotation=45, ha='right')
axes[2].set_yticklabels(ERP_COLS)
axes[2].set_title("Correlation entre composantes ERP")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_signal.png'), dpi=130)
plt.close(fig)

# Paires fortement correlees -> justifie l'etape de reduction du script d'adaptation
cm = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("\n--- Paires de composantes ERP correlees |r| > 0.9 ---")
strong = cm[cm > 0.9].sort_values(ascending=False)
print(strong.round(3).to_string() if len(strong) else "aucune")


# --------------------------------------------------------------------------
# 5. Taille d'effet univariee par Congruence (export pour l'analyse de
#    convergence, point 7)
# --------------------------------------------------------------------------
print("\n--- Cohen's d apparie (par sujet) : Congruence sur composantes ERP ---")
rows = []
for c in ERP_COLS:
    per_sub = glob.groupby(['Sujet_ID', 'Congruence'])[c].mean().unstack()
    if per_sub.shape[1] != 2:
        continue
    diff = (per_sub.iloc[:, 0] - per_sub.iloc[:, 1]).dropna()
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    rows.append({'composante': c, 'cohens_d_apparie': d, 'abs_d': abs(d), 'n_sujets': len(diff)})
eff = pd.DataFrame(rows).sort_values('abs_d', ascending=False)
print(eff.round(4).to_string(index=False))
eff.to_csv(os.path.join(OUT, 'effets_univaries_erp.csv'), index=False)

print(f"\nFigures et effets univaries sauvegardes dans : {os.path.abspath(OUT)}")
