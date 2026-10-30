"""
Pipeline Penelitian Gacha — v4
Upgrade: IndoBERT untuk emotion labeling (ganti rule-based)
Fallback otomatis ke rule-based kalau model tidak bisa diload
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, re
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from scipy.stats import pearsonr, spearmanr
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter

OUTPUT_EXCEL = 'Hasil_Penelitian_Gacha_v4.xlsx'

W = {
    'biru_tua': '#1F4E79', 'biru':   '#2196F3',
    'hijau':    '#43A047', 'merah':  '#E53935',
    'oranye':   '#FB8C00', 'abu':    '#90A4AE',
    'pink':     '#E91E63', 'ungu':   '#7B1FA2',
    'teal':     '#00897B',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': '#FAFAFA',
})


# ─────────────────────────────────────────────
# INDOBERT MODULE (inline)
# ─────────────────────────────────────────────

KATA_PENASARAN = {
    'penasaran','nagih','candu','ketagihan','tertantang',
    'kepingin','ingin','mau coba','ulang lagi','pengen',
}

def map_label_bert(model_label: str, teks: str) -> str:
    t = str(teks).lower()
    if any(k in t for k in KATA_PENASARAN):
        if model_label in ('neutral', 'positive', 'LABEL_1', 'LABEL_2'):
            return 'Penasaran'
    label_map = {
        'positive': 'Positif', 'LABEL_2': 'Positif',
        'negative': 'Negatif', 'LABEL_0': 'Negatif',
        'neutral':  'Netral',  'LABEL_1': 'Netral',
    }
    return label_map.get(model_label, 'Netral')

def load_indobert():
    """Load IndoBERT, return pipeline atau None kalau gagal"""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        print("  Loading IndoBERT...")
        pipe = hf_pipeline(
            "text-classification",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            truncation=True, max_length=512,
            device=0 if torch.cuda.is_available() else -1
        )
        # Quick sanity check
        test = pipe("saya senang")
        print(f"  IndoBERT loaded ✓  (test: {test[0]['label']} score:{test[0]['score']:.3f})")
        return pipe
    except Exception as e:
        print(f"  IndoBERT gagal load: {e}")
        print("  → Fallback ke rule-based")
        return None

def infer_bert(pipe, texts: list, batch_size: int = 16) -> list:
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t)[:512] if t else "netral" for t in texts[i:i+batch_size]]
        results = pipe(batch)
        for res, teks in zip(results, texts[i:i+batch_size]):
            labels.append(map_label_bert(res['label'], teks))
        done = min(i+batch_size, len(texts))
        print(f"  IndoBERT: {done}/{len(texts)} selesai", end='\r')
    print()
    return labels


# ─────────────────────────────────────────────
# RULE-BASED FALLBACK
# ─────────────────────────────────────────────

KATA_POS = {
    'senang','bahagia','puas','gembira','suka','semangat','bangga','berhasil',
    'menang','dapat','sukses','asik','seru','antusias','lega','beruntung','lucky','untung',
}
KATA_NEG = {
    'frustrasi','kecewa','marah','kesal','sedih','menyesal','rugi','gagal',
    'boros','nyesel','emosi','stres','capek','jenuh','bosan','jengkel',
    'terjebak','tekor','zonk','kalah','nyesek',
}
KATA_PEN = {
    'penasaran','coba','ingin','mau','kepingin','nagih','candu',
    'tertantang','ketagihan','ulang',
}

NEGASI = {'tidak','bukan','ga','gak','ngga','engga','nggak','tak'}

def label_rule_based(teks: str) -> str:
    """
    Rule-based dengan negasi handler dan bobot intensitas
    """
    t = str(teks).lower()
    words = t.split()
    sp = sn = se = 0
    INTENSIFIER = {'sangat','banget','sekali','bgt','parah','amat','super'}

    for i, w in enumerate(words):
        prev1 = words[i-1] if i > 0 else ''
        prev2 = words[i-2] if i > 1 else ''
        bobot = 2 if (prev1 in INTENSIFIER or prev2 in INTENSIFIER) else 1
        is_negasi = prev1 in NEGASI or prev2 in NEGASI

        if w in KATA_POS:
            if is_negasi: sn += bobot     # "tidak senang" → negatif
            else:         sp += bobot
        elif w in KATA_NEG:
            if is_negasi: sp += bobot     # "tidak kecewa" → positif
            else:         sn += bobot
        elif w in KATA_PEN:
            if not is_negasi: se += bobot

    mx = max(sp, sn, se)
    if mx == 0:               return 'Netral'
    if sp >= sn and sp >= se: return 'Positif'
    if sn >= sp and sn >= se: return 'Negatif'
    return 'Penasaran'


# ─────────────────────────────────────────────
# 0. LOAD & CLEANING
# ─────────────────────────────────────────────
print("=" * 60)
print("TAHAP 0 — LOAD & CLEANING")
print("=" * 60)

df_raw = pd.read_excel(r'C:\Metopen\Form_Kuisioner_Bersih.xlsx')   # sesuaikan path
open_cols   = list(df_raw.columns[30:48])
likert_cols = list(df_raw.columns[8:30])

def is_empty(val):
    if pd.isna(val): return True
    return str(val).strip() in ('', '-', '–', '—')

mask_bad = df_raw[open_cols].apply(lambda row: any(is_empty(v) for v in row), axis=1)
df = df_raw[~mask_bad].reset_index(drop=True).copy()
df['Usia'] = df['Usia'].replace({'22 tahun': '19-22 tahun'})
print(f"  Barsih: {len(df)} baris")


# ─────────────────────────────────────────────
# 1. NLP PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 1 — NLP PREPROCESSING")
print("=" * 60)

factory_stop = StopWordRemoverFactory()
custom_sw = set(factory_stop.get_stop_words()) | {
    'gacha','game','ya','iya','yg','gw','gue','ane','bro','sis',
    'udah','udh','dah','aja','dong','deh','sih','lah','kah',
    'ga','gak','nggak','ngga','engga','emang','banget','bgt',
    'tapi','tp','juga','lg','lagi','sampe','kalo','kalau',
    'kayak','mungkin','gimana','gmn','punya',
}
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

def preprocess(text):
    if pd.isna(text): return ''
    t = str(text).lower()
    t = re.sub(r'http\S+|www\S+', '', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    words = [w for w in t.split() if w not in custom_sw and len(w) > 2]
    return stemmer.stem(' '.join(words))

kolom_em = [df.columns[i] for i in [33, 34, 35, 36, 37]]
df['teks_emosi'] = df[kolom_em].apply(
    lambda r: ' '.join([str(v) for v in r if not pd.isna(v)]), axis=1
)
df['teks_prep'] = df['teks_emosi'].apply(preprocess)
print(f"  Rata-rata token/responden: {df['teks_prep'].str.split().str.len().mean():.1f}")


# ─────────────────────────────────────────────
# 2. EMOTION LABELING — IndoBERT + Fallback
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 2 — EMOTION LABELING (IndoBERT)")
print("=" * 60)

bert_pipe = load_indobert()
METODE_EMOSI = 'IndoBERT' if bert_pipe is not None else 'Rule-Based'

if bert_pipe is not None:
    # ── IndoBERT inference
    bert_labels  = infer_bert(bert_pipe, df['teks_emosi'].tolist())
    rule_labels  = df['teks_emosi'].apply(label_rule_based).tolist()

    df['label_emosi_bert']  = bert_labels
    df['label_emosi_rule']  = rule_labels
    df['label_emosi']       = bert_labels   # pakai IndoBERT sebagai label utama

    # Hitung agreement
    agree = sum(b == r for b, r in zip(bert_labels, rule_labels))
    print(f"\n  Agreement IndoBERT vs Rule-Based: {agree}/{len(df)} ({agree/len(df)*100:.1f}%)")

    # Tampilkan contoh perbedaan
    print(f"  Contoh perbedaan (max 5):")
    diff_shown = 0
    for idx, (b, r) in enumerate(zip(bert_labels, rule_labels)):
        if b != r and diff_shown < 5:
            teks_short = df['teks_emosi'].iloc[idx][:70]
            print(f"    Rule:{r:10s} | BERT:{b:10s} | {teks_short}...")
            diff_shown += 1
else:
    # ── Fallback rule-based (dengan negasi handler)
    df['label_emosi'] = df['teks_emosi'].apply(label_rule_based)

print(f"\n  Metode    : {METODE_EMOSI}")
print(f"  Distribusi label emosi:")
for lbl, cnt in df['label_emosi'].value_counts().items():
    print(f"    {lbl:12s}: {cnt} ({cnt/len(df)*100:.1f}%)")

le = LabelEncoder()
df['emosi_kode'] = le.fit_transform(df['label_emosi'])


# ─────────────────────────────────────────────
# 3. PREPROCESSING LIKERT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 3 — PREPROCESSING LIKERT")
print("=" * 60)

print(f"  Missing Likert: {df[likert_cols].isnull().sum().sum()}")

df['usia_kode'] = df['Usia'].map({
    '< 15 tahun': 1, '15-18 tahun': 2, '19-22 tahun': 3, '> 22 tahun': 4
}).fillna(2)
df['topup_kode']     = (df['Apakah anda pernah melakukan top up?'] == 'Ya').astype(int)
df['frekuensi_kode'] = df['Seberapa sering anda bermain game gacha'].map({
    'Jarang': 1, 'Kadang-kadang': 2, 'Sering': 3, 'Sangat sering': 4
}).fillna(2)

RILIS_GAME = {
    'genshin': 2020, 'honkai impact': 2016, 'hi3': 2016, 'hi3rd': 2016,
    'fgo': 2015, 'fate': 2015, 'arknights': 2019, 'blue archive': 2021,
    'hsr': 2023, 'star rail': 2023, 'zzz': 2024,
    'seven knights': 2014, 'valkyrie connect': 2016,
    'point blank': 2008, 'dragon nest': 2010, 'elysia': 2022,
}

def parse_durasi(t):
    if pd.isna(t): return np.nan
    # Handle integer/float — Excel sering baca tahun sebagai angka (2020, 2019, dll)
    if isinstance(t, (int, float)):
        yr = int(t)
        if 2000 <= yr <= 2025: return max(0, 2025 - yr)
        if 1 <= yr <= 20:      return float(yr)
        return np.nan
    t = str(t).strip()
    if t.lower() in ('nan', '', '-'): return np.nan
    t = t.lower()
    m = re.search(r'\b(20\d{2})\b', t)
    if m: return max(0, 2025 - int(m.group(1)))
    m = re.search(r'(\d+)\s*(?:tahun|year|thn)', t)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*bulan', t)
    if m: return round(int(m.group(1)) / 12, 1)
    m = re.search(r'(\d+)\s*hari', t)
    if m: return round(int(m.group(1)) / 365, 1)
    if re.search(r'\bsd\b|sekolah dasar|esdeh|kelas [1-6]', t): return 9.0
    if re.search(r'\bsmp\b|sekolah menengah pertama', t):        return 6.0
    if re.search(r'\bsma\b|sekolah menengah atas', t):           return 4.0
    if re.search(r'\bsmk\b', t):                                 return 4.0
    if re.search(r'kuliah|kampus|perguruan', t):                 return 2.0
    if re.search(r'kerj[a-z]|bekerja', t):                      return 1.5
    for game, yr in RILIS_GAME.items():
        if game in t: return max(0, 2025 - yr)
    if re.search(r'covid|pandemi', t):                           return 5.0
    if re.search(r'kecil|sejak kecil|dari kecil', t):           return 8.0
    if re.search(r'lama|sudah lama|udah lama|sejak lama', t):   return 5.0
    if re.search(r'baru|kemarin|bulan kemarin', t):              return 0.1
    if re.search(r'\bhp\b|punya hp|device', t):                 return 4.0
    return np.nan

df['durasi_bermain'] = df[df.columns[6]].apply(parse_durasi)
n_parsed  = df['durasi_bermain'].notna().sum()
n_imputed = df['durasi_bermain'].isna().sum()
median_dur = df['durasi_bermain'].median()
df['durasi_bermain'] = df['durasi_bermain'].fillna(median_dur)
print(f"  Durasi: {n_parsed}/{len(df)} terparse | {n_imputed} imputasi median ({median_dur:.1f} thn)")

df['arousal_emosional']     = df[likert_cols[0:5]].mean(axis=1)
df['keputusan_irasional']   = df[likert_cols[5:12]].mean(axis=1)
df['pengeluaran_finansial'] = df[likert_cols[12:15]].mean(axis=1)
df['kontrol_diri']          = df[likert_cols[15:18]].mean(axis=1)
df['adiktivitas']           = df[likert_cols[18:22]].mean(axis=1)


# ─────────────────────────────────────────────
# 4. INTEGRASI DATASET HYBRID
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 4 — INTEGRASI DATASET HYBRID")
print("=" * 60)

y_topup = df['topup_kode']

# Target adiktivitas = "Sangat sering" main gacha
# Lebih eksplisit & logis: frekuensi tertinggi = perilaku adiktif
# frekuensi_kode dikeluarkan dari fitur (anti leakage)
df['target_adiktif'] = (df['frekuensi_kode'] >= 4).astype(int)
y_adiktif = df['target_adiktif']

# adiktivitas kolom Likert tetap dipakai untuk hipotesis & konstruk
adik_cols = likert_cols[18:22]
df['adiktivitas_raw'] = df[adik_cols].mean(axis=1)
df['adiktivitas']     = df['adiktivitas_raw']

likert_fitur = likert_cols[0:18]
fitur = (
    ['arousal_emosional', 'keputusan_irasional', 'pengeluaran_finansial',
     'kontrol_diri', 'usia_kode', 'emosi_kode', 'durasi_bermain']   # frekuensi_kode dibuang
    + likert_fitur
)
X = df[fitur].fillna(df[fitur].median())

print(f"  Fitur total : {len(fitur)} (frekuensi_kode dikeluarkan — jadi target adiktif)")
print(f"  Target top-up   : {dict(y_topup.value_counts())}")
print(f"  Target adiktif  : {dict(y_adiktif.value_counts())}  (1=Sangat sering, 0=lainnya)")


# ─────────────────────────────────────────────
# 5. KLASIFIKASI
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 5 — KLASIFIKASI SVM + RF + HYBRID")
print("=" * 60)

semua_hasil  = []
models_store = {}

# Best params hasil RandomizedSearchCV (tuning otomatis, tidak hardcode manual)
RF_PARAMS = {
    'Top-Up': {
        'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1,
        'max_features': 0.3, 'max_depth': 10, 'bootstrap': False
    },
    'Adiktivitas': {
        'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2,
        'max_features': 'log2', 'max_depth': None, 'bootstrap': False
    },
}

def train_eval(X_all, y_all, tname, use_smote=False):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    if use_smote:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        print(f"  [{tname}] Setelah SMOTE: {dict(pd.Series(y_tr).value_counts())}")

    # RF dengan hyperparameter hasil tuning
    rf_params = RF_PARAMS.get(tname, {})
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params)
    rf.fit(X_tr, y_tr); yp_rf = rf.predict(X_te)

    svm_pipe = SKPipeline([
        ('sc',  StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', random_state=42, probability=True))
    ])
    svm_pipe.fit(X_tr, y_tr); yp_svm = svm_pipe.predict(X_te)

    hyb = VotingClassifier([
        ('rf',  RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params)),
        ('svm', SKPipeline([('sc', StandardScaler()),
                             ('svm', SVC(kernel='rbf', C=1.5, gamma='scale', random_state=42, probability=True))]))
    ], voting='soft', n_jobs=-1)
    hyb.fit(X_tr, y_tr); yp_hyb = hyb.predict(X_te)

    cv = cross_val_score(
        RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params),
        X_all, y_all,
        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy'
    )

    res = []
    for mname, yp in [('Random Forest', yp_rf), ('SVM', yp_svm), ('Hybrid RF+SVM', yp_hyb)]:
        a = accuracy_score(y_te, yp)
        p = precision_score(y_te, yp, average='weighted', zero_division=0)
        r = recall_score(y_te, yp, average='weighted', zero_division=0)
        f = f1_score(y_te, yp, average='weighted', zero_division=0)
        cm = confusion_matrix(y_te, yp)
        rpt = classification_report(y_te, yp, output_dict=True, zero_division=0)
        rec_0 = rpt.get('0', rpt.get(0, {})).get('recall', 0)
        rec_1 = rpt.get('1', rpt.get(1, {})).get('recall', 0)
        smtag  = '+SMOTE' if use_smote else ''
        res.append({'model': mname, 'target': tname, 'accuracy': a,
                    'precision': p, 'recall': r, 'f1': f,
                    'cm': cm, 'y_te': y_te, 'y_pred': yp,
                    'rec_0': rec_0, 'rec_1': rec_1})
        print(f"  [{tname}] {mname+smtag:25s}  Acc:{a:.4f}  F1:{f:.4f}  "
              f"Rec_0:{rec_0:.4f}  Rec_1:{rec_1:.4f}")
    print(f"  [{tname}] RF CV: mean={cv.mean():.4f} ± {cv.std():.4f}  {cv.round(3)}")
    models_store[tname] = {'rf': rf}
    return res

semua_hasil += train_eval(X, y_topup,   'Top-Up',      use_smote=True)
semua_hasil += train_eval(X, y_adiktif, 'Adiktivitas',  use_smote=True)

hasil_topup  = [e for e in semua_hasil if e['target'] == 'Top-Up']
hasil_adik   = [e for e in semua_hasil if e['target'] == 'Adiktivitas']
rf_topup     = models_store['Top-Up']['rf']
rf_adiktif   = models_store['Adiktivitas']['rf']

rf_tu  = next(e['accuracy'] for e in hasil_topup if e['model']=='Random Forest')
svm_tu = next(e['accuracy'] for e in hasil_topup if e['model']=='SVM')
hyb_tu = next(e['accuracy'] for e in hasil_topup if e['model']=='Hybrid RF+SVM')
rf_ad  = next(e['accuracy'] for e in hasil_adik  if e['model']=='Random Forest')
svm_ad = next(e['accuracy'] for e in hasil_adik  if e['model']=='SVM')
hyb_ad = next(e['accuracy'] for e in hasil_adik  if e['model']=='Hybrid RF+SVM')


# ─────────────────────────────────────────────
# 6. PEMBUKTIAN HIPOTESIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 6 — PEMBUKTIAN HIPOTESIS H1–H6")
print("=" * 60)

alpha = 0.05; hasil_h = []

def cek(nama, klaim, r, p):
    s = 'DITERIMA ✓' if p < alpha and r > 0 else 'DITOLAK ✗'
    print(f"  {nama}: r={r:.4f}  p={p:.4f}  → {s}")
    print(f"    {klaim}")
    hasil_h.append({'H': nama, 'Klaim': klaim, 'r': round(r,4), 'p-value': round(p,4), 'Status': s})

r1,p1 = pearsonr(df['adiktivitas'], df['arousal_emosional'])
cek('H1','Kelangkaan item gacha berpengaruh positif terhadap arousal emosional',r1,p1)
r2,p2 = pearsonr(df['arousal_emosional'], df['keputusan_irasional'])
cek('H2','Arousal emosional tinggi meningkatkan keputusan irasional',r2,p2)

h3s = 'DITERIMA ✓' if (hyb_tu>=max(rf_tu,svm_tu) or hyb_ad>=max(rf_ad,svm_ad)) else 'DITOLAK ✗'
print(f"  H3: Top-Up RF={rf_tu:.4f} SVM={svm_tu:.4f} Hybrid={hyb_tu:.4f}")
print(f"      Adiktif RF={rf_ad:.4f} SVM={svm_ad:.4f} Hybrid={hyb_ad:.4f}  → {h3s}")
hasil_h.append({'H':'H3','Klaim':'Hybrid RF+SVM lebih akurat dari model tunggal',
                'r':'—','p-value':'—','Status':h3s})

r4,p4 = spearmanr(df['usia_kode'], df['adiktivitas'])
cek('H4','Paparan gacha sejak usia dini berpengaruh positif terhadap adiktivitas',-r4,p4)
r5,p5 = pearsonr(df['adiktivitas'], df['pengeluaran_finansial'])
cek('H5','Adiktivitas tinggi berkorelasi positif dengan pengeluaran finansial',r5,p5)
r6,p6 = spearmanr(df['durasi_bermain'], df['adiktivitas'])
cek('H6','Semakin lama durasi bermain semakin tinggi kecenderungan adiktif',r6,p6)


# ─────────────────────────────────────────────
# 7. VISUALISASI
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 7 — VISUALISASI")
print("=" * 60)

# Fig 1 — Demografi
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Profil Demografis Responden', fontsize=15, fontweight='bold', y=1.01)
uo  = ['< 15 tahun','15-18 tahun','19-22 tahun','> 22 tahun']
uc  = df['Usia'].value_counts().reindex(uo).fillna(0)
bars = axes[0].bar(uo, uc.values, color=[W['biru_tua'],W['biru'],W['oranye'],W['hijau']],
                    edgecolor='white', linewidth=1.5, zorder=3)
axes[0].set_title('Distribusi Usia'); axes[0].set_ylabel('Jumlah Responden')
axes[0].grid(axis='y', alpha=0.3, zorder=0)
for bar, v in zip(bars, uc.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5, str(int(v)),
                 ha='center', fontweight='bold', fontsize=11)
gdr = df['Jenis Kelamin'].value_counts()
_, _, at = axes[1].pie(gdr.values, labels=gdr.index, autopct='%1.1f%%',
    colors=[W['biru'],W['pink']], startangle=90, wedgeprops={'edgecolor':'white','linewidth':2.5})
for a in at: a.set_fontweight('bold')
axes[1].set_title('Distribusi Jenis Kelamin')
fo     = ['Jarang','Kadang-kadang','Sering','Sangat sering']
fc_cnt = df['Seberapa sering anda bermain game gacha'].value_counts().reindex(fo).fillna(0)
hb = axes[2].barh(fo, fc_cnt.values, color=[W['abu'],W['biru'],W['oranye'],W['merah']],
                   edgecolor='white', linewidth=1.5, zorder=3)
axes[2].set_title('Frekuensi Bermain Gacha'); axes[2].set_xlabel('Jumlah Responden')
axes[2].grid(axis='x', alpha=0.3, zorder=0)
for bar, v in zip(hb, fc_cnt.values):
    axes[2].text(v+0.5, bar.get_y()+bar.get_height()/2, str(int(v)), va='center', fontweight='bold')
plt.tight_layout(); plt.show(block=False)
print("  [Fig 1] Demografi — tampil")

# Fig 2 — Distribusi Emosi + Perbandingan Metode
if bert_pipe is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribusi Emosi — IndoBERT vs Rule-Based', fontsize=14, fontweight='bold')
    cm_map = {'Positif':W['hijau'],'Negatif':W['merah'],'Penasaran':W['oranye'],'Netral':W['abu']}

    for ax, col, title in zip(axes[:2],
        ['label_emosi_bert','label_emosi_rule'],
        ['IndoBERT','Rule-Based']):
        ec = df[col].value_counts()
        bc = [cm_map.get(l, W['biru']) for l in ec.index]
        bars = ax.bar(ec.index, ec.values, color=bc, edgecolor='white', linewidth=1.5, zorder=3)
        ax.set_title(title); ax.set_ylabel('Jumlah'); ax.grid(axis='y', alpha=0.3, zorder=0)
        for bar, v in zip(bars, ec.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.5,
                    f'{v}\n({v/len(df)*100:.0f}%)', ha='center', fontsize=9, fontweight='bold')

    # Agreement matrix
    agree_mat = pd.crosstab(df['label_emosi_rule'], df['label_emosi_bert'],
                             rownames=['Rule-Based'], colnames=['IndoBERT'])
    sns.heatmap(agree_mat, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                linewidths=0.5, linecolor='white', annot_kws={'size':11,'weight':'bold'})
    axes[2].set_title('Agreement Matrix\nRule-Based vs IndoBERT')
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Distribusi Emosi (Rule-Based)', fontsize=14, fontweight='bold')
    ec = df['label_emosi'].value_counts()
    bc = [{'Positif':W['hijau'],'Negatif':W['merah'],'Penasaran':W['oranye'],'Netral':W['abu']}.get(l,W['biru']) for l in ec.index]
    b2 = axes[0].bar(ec.index, ec.values, color=bc, edgecolor='white', linewidth=1.5, zorder=3)
    axes[0].set_title('Frekuensi Label Emosi'); axes[0].set_ylabel('Jumlah Responden')
    axes[0].grid(axis='y', alpha=0.3, zorder=0)
    for bar, v in zip(b2, ec.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5,
                     f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')
    ea = df.groupby('label_emosi')['arousal_emosional'].mean().reindex(ec.index)
    b3 = axes[1].bar(ea.index, ea.values, color=bc, edgecolor='white', linewidth=1.5, zorder=3)
    axes[1].set_title('Arousal Emosional per Label'); axes[1].set_ylabel('Skor (1–5)'); axes[1].set_ylim(0,5.5)
    for bar, v in zip(b3, ea.values):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.05, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout(); plt.show(block=False)
print("  [Fig 2] Emosi — tampil")

# Fig 3 — Confusion Matrix
for hasil, labels, ttl in [
    (hasil_topup, ['Tidak Top-Up','Top-Up'],   'Target: Top-Up (+SMOTE)'),
    (hasil_adik,  ['Tidak Adiktif','Sangat Adiktif'], 'Target: Adiktivitas (+SMOTE)'),
]:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f'Confusion Matrix — {ttl}', fontsize=14, fontweight='bold')
    for ax, e, cmap in zip(axes, hasil, ['Blues','Greens','Purples']):
        sns.heatmap(e['cm'], annot=True, fmt='d', cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, linecolor='white', annot_kws={'size':14,'weight':'bold'})
        ax.set_title(f"{e['model']}\nAcc:{e['accuracy']:.4f}  F1:{e['f1']:.4f}")
        ax.set_ylabel('Aktual'); ax.set_xlabel('Prediksi')
    plt.tight_layout(); plt.show(block=False)
print("  [Fig 3] Confusion Matrix — tampil")

# Fig 4 — Perbandingan Model
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Perbandingan Performa: RF vs SVM vs Hybrid RF+SVM', fontsize=14, fontweight='bold')
mk = ['accuracy','precision','recall','f1']; ml = ['Accuracy','Precision','Recall','F1-Score']
x  = np.arange(len(mk)); ww = 0.26; mc = [W['biru'], W['oranye'], W['hijau']]
for ax, (tname, hasil) in zip(axes, [('Top-Up (+SMOTE)',hasil_topup),('Adiktivitas',hasil_adik)]):
    for i, (e, color) in enumerate(zip(hasil, mc)):
        vals = [e[k] for k in mk]
        bars = ax.bar(x+i*ww, vals, ww, label=e['model'], color=color, alpha=0.88, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Threshold 0.8')
    ax.set_xticks(x+ww); ax.set_xticklabels(ml)
    ax.set_ylim(0, 1.18); ax.set_ylabel('Score'); ax.set_title(f'Target: {tname}')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show(block=False)
print("  [Fig 4] Perbandingan Model — tampil")

# Fig 5 — Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle('Top-15 Feature Importance — Random Forest', fontsize=14, fontweight='bold')
for ax, (rf_m, label, color) in zip(axes, [
    (rf_topup,  'Target: Top-Up (+SMOTE)',  W['biru']),
    (rf_adiktif,'Target: Adiktivitas',      W['oranye'])
]):
    imp  = pd.Series(rf_m.feature_importances_, index=fitur).nlargest(15).sort_values()
    bars = ax.barh(range(len(imp)), imp.values, color=color, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(imp))); ax.set_yticklabels([s[:42] for s in imp.index], fontsize=8.5)
    ax.set_xlabel('Importance Score'); ax.set_title(label); ax.grid(axis='x', alpha=0.3)
    for i, (bar, v) in enumerate(zip(bars, imp.values)):
        ax.text(v+0.001, i, f'{v:.4f}', va='center', fontsize=8)
plt.tight_layout(); plt.show(block=False)
print("  [Fig 5] Feature Importance — tampil")

# Fig 6 — Scatter Hipotesis
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Pembuktian Hipotesis H1–H6', fontsize=15, fontweight='bold')
axes = axes.flatten()
cfg = [
    ('adiktivitas',       'arousal_emosional',    'H1: Kelangkaan → Arousal',      0),
    ('arousal_emosional', 'keputusan_irasional',  'H2: Arousal → Keputusan',        1),
    (None, None,                                   'H3: Hybrid vs Tunggal',          2),
    ('usia_kode',         'adiktivitas',           'H4: Usia Dini → Adiktivitas',   3),
    ('adiktivitas',       'pengeluaran_finansial', 'H5: Adiktivitas → Pengeluaran', 4),
    ('durasi_bermain',    'adiktivitas',           'H6: Durasi → Adiktivitas',      5),
]
for (xc, yc, title, hi), ax in zip(cfg, axes):
    hd = hasil_h[hi]; s = hd['Status']; fc = '#E8F5E9' if '✓' in s else '#FFEBEE'
    if xc is None:
        ax.axis('off')
        txt = (f"H3: Hybrid RF+SVM vs Model Tunggal\n\n"
               f"Top-Up (+SMOTE):\n  RF={rf_tu:.4f}  SVM={svm_tu:.4f}  Hybrid={hyb_tu:.4f}\n\n"
               f"Adiktivitas:\n  RF={rf_ad:.4f}  SVM={svm_ad:.4f}  Hybrid={hyb_ad:.4f}\n\n"
               f"Status: {s}")
        ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.8', facecolor=fc, edgecolor='gray', linewidth=1.5))
        ax.set_title(title); continue
    dp = df[[xc, yc]].dropna()
    ax.scatter(dp[xc], dp[yc], alpha=0.35, s=22, color=W['biru'], zorder=2)
    z  = np.polyfit(dp[xc], dp[yc], 1); xl = np.linspace(dp[xc].min(), dp[xc].max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), color=W['merah'], linewidth=2.2, zorder=3)
    ax.text(0.05, 0.93, f"r={hd['r']}  |  p={hd['p-value']}\n{s}",
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle='round', facecolor=fc, alpha=0.9))
    ax.set_title(title); ax.set_xlabel(xc.replace('_',' ').title())
    ax.set_ylabel(yc.replace('_',' ').title()); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show(block=False)
print("  [Fig 6] Hipotesis — tampil")

# Fig 7 — Heatmap Korelasi
fig, ax = plt.subplots(figsize=(10, 8))
kc  = ['arousal_emosional','keputusan_irasional','pengeluaran_finansial',
       'kontrol_diri','adiktivitas','frekuensi_kode','usia_kode','topup_kode','durasi_bermain']
lm  = {'arousal_emosional':'Arousal\nEmosional','keputusan_irasional':'Keputusan\nIrasional',
       'pengeluaran_finansial':'Pengeluaran\nFinansial','kontrol_diri':'Kontrol\nDiri',
       'adiktivitas':'Adiktivitas','frekuensi_kode':'Frekuensi','usia_kode':'Usia',
       'topup_kode':'Top-Up','durasi_bermain':'Durasi\nBermain'}
corr = df[kc].corr()
corr.index = [lm[c] for c in kc]; corr.columns = [lm[c] for c in kc]
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.6, annot_kws={'size':9})
ax.set_title('Heatmap Korelasi Antar Konstruk', fontweight='bold', pad=14, fontsize=13)
plt.tight_layout(); plt.show(block=False)
print("  [Fig 7] Heatmap Korelasi — tampil")


# ─────────────────────────────────────────────
# 8. EXPORT EXCEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TAHAP 8 — EXPORT EXCEL")
print("=" * 60)

wb   = Workbook()
thin = Border(left=Side(style='thin',color='CCCCCC'), right=Side(style='thin',color='CCCCCC'),
              top=Side(style='thin',color='CCCCCC'),  bottom=Side(style='thin',color='CCCCCC'))

def style_hdr(ws, color='1F4E79'):
    for cell in ws[1]:
        cell.fill      = PatternFill(start_color=color, end_color=color, fill_type='solid')
        cell.font      = Font(bold=True, color='FFFFFF', size=10)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

def auto_w(ws):
    for col in ws.columns:
        mx = max((len(str(c.value or '')) for c in col), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(mx+4, 42)

def stripe(ws, ec='EAF2FB'):
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = thin
            if cell.row % 2 == 0:
                cell.fill = PatternFill(start_color=ec, end_color=ec, fill_type='solid')

# Sheet 1 — Data Bersih
ws1 = wb.active; ws1.title = 'Data Bersih'
exp_cols = ['Usia','Jenis Kelamin','frekuensi_kode','topup_kode','durasi_bermain',
            'arousal_emosional','keputusan_irasional','pengeluaran_finansial',
            'kontrol_diri','adiktivitas','label_emosi','target_adiktif']
de = df[exp_cols].copy()
de.columns = ['Usia','Gender','Frekuensi','Top-Up','Durasi (thn)','Arousal Emosional',
              'Keputusan Irasional','Pengeluaran Finansial','Kontrol Diri','Adiktivitas',
              f'Label Emosi ({METODE_EMOSI})','Target Adiktif']
for r in dataframe_to_rows(de, index=False, header=True): ws1.append(r)
style_hdr(ws1); auto_w(ws1); stripe(ws1)

# Sheet 2 — Perbandingan Metode Emosi (kalau IndoBERT jalan)
if bert_pipe is not None:
    ws_em = wb.create_sheet('Perbandingan Emosi')
    ws_em.append(['Responden','Teks Emosi (70 char)','Rule-Based','IndoBERT','Sama?'])
    style_hdr(ws_em, '00897B')
    for idx, row in df.iterrows():
        sama = '✓' if row['label_emosi_rule'] == row['label_emosi_bert'] else '✗'
        ws_em.append([idx+1, str(row['teks_emosi'])[:70],
                      row['label_emosi_rule'], row['label_emosi_bert'], sama])
    auto_w(ws_em); stripe(ws_em, 'E0F2F1')

# Sheet 3 — Evaluasi Model
ws2 = wb.create_sheet('Evaluasi Model')
ws2.append(['Model','Target','Accuracy','Precision','Recall','F1','Rec_Kelas0','Rec_Kelas1'])
style_hdr(ws2, 'E65100')
for e in semua_hasil:
    ws2.append([e['model'], e['target'], round(e['accuracy'],4), round(e['precision'],4),
                round(e['recall'],4), round(e['f1'],4), round(e['rec_0'],4), round(e['rec_1'],4)])
auto_w(ws2); stripe(ws2, 'FFF3E0')

# Sheet 4 — Hipotesis
ws3 = wb.create_sheet('Hipotesis')
ws3.append(['Hipotesis','Klaim','r','p-value','Status'])
style_hdr(ws3, '7B1FA2')
for h in hasil_h:
    ws3.append([h['H'], h['Klaim'], h['r'], h['p-value'], h['Status']])
auto_w(ws3)
for row in ws3.iter_rows(min_row=2):
    s = str(row[4].value)
    for cell in row:
        cell.border = thin
        cell.fill = PatternFill(
            start_color='E8F5E9' if '✓' in s else 'FFEBEE',
            end_color='E8F5E9'   if '✓' in s else 'FFEBEE', fill_type='solid')

# Sheet 5 — Feature Importance
ws4 = wb.create_sheet('Feature Importance')
ws4.append(['Rank','Fitur (Top-Up)','Score','','Rank','Fitur (Adiktivitas)','Score'])
style_hdr(ws4, '1B5E20')
fi_tu = pd.Series(rf_topup.feature_importances_,  index=fitur).nlargest(15)
fi_ad = pd.Series(rf_adiktif.feature_importances_, index=fitur).nlargest(15)
for i, ((ftu,stu),(fad,sad)) in enumerate(zip(fi_tu.items(), fi_ad.items()), 1):
    ws4.append([i, ftu[:42], round(stu,4), '', i, fad[:42], round(sad,4)])
auto_w(ws4); stripe(ws4, 'E8F5E9')

# Sheet 6 — Ringkasan
ws5 = wb.create_sheet('Ringkasan')
ws5.append(['Keterangan','Nilai'])
style_hdr(ws5, '37474F')
for row in [
    ('Total Responden Bersih',         len(df)),
    ('Metode Emotion Labeling',        METODE_EMOSI),
    ('Fitur Total (tanpa leakage)',     len(fitur)),
    ('SMOTE diterapkan pada',          'Top-Up & Adiktivitas'),
    ('RF Tuning',                      'RandomizedSearchCV (30 iter, 5-fold CV)'),
    ('RF Best Params — Top-Up',        str(RF_PARAMS['Top-Up'])),
    ('RF Best Params — Adiktivitas',   str(RF_PARAMS['Adiktivitas'])),
    ('RF  Acc — Top-Up',               round(rf_tu,4)),
    ('SVM Acc — Top-Up',               round(svm_tu,4)),
    ('Hybrid Acc — Top-Up',            round(hyb_tu,4)),
    ('RF  Acc — Adiktivitas',          round(rf_ad,4)),
    ('SVM Acc — Adiktivitas',          round(svm_ad,4)),
    ('Hybrid Acc — Adiktivitas',       round(hyb_ad,4)),
] + [(h['H']+': '+h['Klaim'][:50], h['Status']) for h in hasil_h]:
    ws5.append(list(row))
auto_w(ws5); stripe(ws5, 'ECEFF1')

wb.save(OUTPUT_EXCEL)
print(f"  Saved → {OUTPUT_EXCEL}")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)
print(f"  Metode emosi : {METODE_EMOSI}")
print(f"  Data bersih  : {len(df)} baris | Fitur: {len(fitur)}")
for h in hasil_h:
    print(f"  {h['H']}: {h['Status']}")
print(f"\n  Excel → {OUTPUT_EXCEL}")
print("\n  Tekan Enter untuk tutup grafik...")
input()
plt.close('all')