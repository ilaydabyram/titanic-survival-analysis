import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
raw = pd.read_csv(src)

# ---------- TEMİZLİK ----------
df = raw.copy()
# 'alive' = survived'in metin hali (sızıntı); who/adult_male/class/embark_town/alone başka sütunlardan türetilmiş
# deck %77 eksik -> atılır
df = df.drop(columns=["alive", "who", "adult_male", "class", "embark_town", "alone", "deck"])
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
age_missing = df["age"].isna()
df["age_filled"] = df["age"].fillna(df.groupby(["pclass", "sex"])["age"].transform("median"))
df["family_size"] = df["sibsp"] + df["parch"] + 1
df.to_csv(os.path.join(outdir, "titanic_clean.csv"), index=False)
print("Temiz veri:", df.shape, "| eksik kalan:", int(df.drop(columns='age').isna().sum().sum()))
print("Yaşı doldurulan yolcu:", int(age_missing.sum()))

# ---------- KEŞİF TABLOLARI ----------
def rate(g):  # hayatta kalma oranı ve yolcu sayısı
    return pd.Series({"oran": g["survived"].mean(), "n": len(g)})
print("\nGenel:", round(df["survived"].mean(), 3))
t_sex = df.groupby("sex").apply(rate, include_groups=False); print("\nCinsiyete göre\n", t_sex.round(3))
t_cls = df.groupby("pclass").apply(rate, include_groups=False); print("\nSınıfa göre\n", t_cls.round(3))
t_sc = df.groupby(["pclass", "sex"]).apply(rate, include_groups=False); print("\nSınıf x cinsiyet\n", t_sc.round(3))
bins = [0, 12, 18, 35, 60, 100]; labels = ["0-12", "13-18", "19-35", "36-60", "60+"]
known = df[~age_missing].copy()
known["yas_grubu"] = pd.cut(known["age"], bins=bins, labels=labels)
t_age = known.groupby("yas_grubu", observed=True).apply(rate, include_groups=False); print("\nYaş grubuna göre (yaşı bilinenler)\n", t_age.round(3))
fam = df["family_size"].clip(upper=5).map(lambda x: "5+" if x >= 5 else str(x))
t_fam = df.assign(aile=fam).groupby("aile").apply(rate, include_groups=False); print("\nAile büyüklüğüne göre\n", t_fam.round(3))
t_emb = df.groupby("embarked").apply(rate, include_groups=False); print("\nBinilen limana göre\n", t_emb.round(3))

# ---------- GRAFİKLER ----------
BG, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": MUTED,
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def style(ax):
    ax.set_ylim(0, 1.0); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8); ax.set_axisbelow(True)

# Grafik 1: sınıf x cinsiyet
fig, ax = plt.subplots(figsize=(7.5, 4.6))
w = 0.34; x = np.arange(3)
for i, (sx, col, lab) in enumerate([("female", BLUE, "Kadın"), ("male", ORANGE, "Erkek")]):
    vals = [t_sc.loc[(c, sx), "oran"] for c in (1, 2, 3)]
    bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=col, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
ax.set_xticks(x); ax.set_xticklabels(["1. sınıf", "2. sınıf", "3. sınıf"])
style(ax); ax.set_ylabel("Hayatta kalma oranı")
ax.set_title("Hayatta kalma: bilet sınıfı ve cinsiyet", loc="left", color=INK, fontsize=13)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(outdir, "g1_sinif_cinsiyet.png"), dpi=160); plt.close(fig)

# Grafik 2: yaş grubu
fig, ax = plt.subplots(figsize=(7.5, 4.4))
v = t_age["oran"].values
bars = ax.bar(labels, v, 0.5, color=BLUE)
for b, val, n in zip(bars, v, t_age["n"]):
    ax.text(b.get_x() + b.get_width()/2, val + 0.02, f"{val:.0%}", ha="center", fontsize=10)
    ax.text(b.get_x() + b.get_width()/2, 0.02, f"n={int(n)}", ha="center", fontsize=9, color="white")
style(ax); ax.set_xlabel("Yaş grubu"); ax.set_ylabel("Hayatta kalma oranı")
ax.set_title("Hayatta kalma: yaş grubuna göre", loc="left", color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "g2_yas.png"), dpi=160); plt.close(fig)

# Grafik 3: aile büyüklüğü
fig, ax = plt.subplots(figsize=(7.5, 4.4))
order = ["1", "2", "3", "4", "5+"]; v = [t_fam.loc[o, "oran"] for o in order]; ns = [t_fam.loc[o, "n"] for o in order]
bars = ax.bar(order, v, 0.5, color=BLUE)
for b, val, n in zip(bars, v, ns):
    ax.text(b.get_x() + b.get_width()/2, val + 0.02, f"{val:.0%}", ha="center", fontsize=10)
    ax.text(b.get_x() + b.get_width()/2, 0.02, f"n={int(n)}", ha="center", fontsize=9, color="white")
style(ax); ax.set_xlabel("Aile büyüklüğü (kendisi dahil)"); ax.set_ylabel("Hayatta kalma oranı")
ax.set_title("Hayatta kalma: aile büyüklüğüne göre", loc="left", color=INK, fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(outdir, "g3_aile.png"), dpi=160); plt.close(fig)
print("\nGrafikler kaydedildi.")
