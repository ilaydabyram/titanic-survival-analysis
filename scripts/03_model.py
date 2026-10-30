import sys, os
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

df = pd.read_csv(sys.argv[1])
y = df["survived"]
X = pd.get_dummies(df[["pclass", "sex", "age_filled", "family_size", "fare", "embarked"]], columns=["sex", "embarked"], drop_first=True).astype(float)
X["pclass"] = X["pclass"].astype(float)
print("Özellikler:", list(X.columns))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Eğitim: {len(X_tr)}  Test: {len(X_te)}  (test'te hayatta kalma oranı {y_te.mean():.3f})")

models = {
    "Baz çizgi (hep 'öldü' de)": DummyClassifier(strategy="most_frequent"),
    "Lojistik regresyon": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Karar ağacı (derinlik 3)": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Karar ağacı (derinlik 8)": DecisionTreeClassifier(max_depth=8, random_state=42),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    cvs = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
    m.fit(X_tr, y_tr)
    tr = accuracy_score(y_tr, m.predict(X_tr)); te = accuracy_score(y_te, m.predict(X_te))
    rows.append({"model": name, "eğitim": tr, "çapraz-doğr. (eğitim)": cvs.mean(), "test": te})
res = pd.DataFrame(rows).set_index("model").round(3)
print("\n", res)

# Seçilen modeller için ayrıntı
for name in ["Lojistik regresyon", "Karar ağacı (derinlik 3)"]:
    p = models[name].predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, p).ravel()
    print(f"\n{name}: kesinlik={precision_score(y_te,p):.3f} duyarlılık={recall_score(y_te,p):.3f}  [TN={tn} FP={fp} FN={fn} TP={tp}]")

# Lojistik regresyon katsayıları (standartlaştırılmış)
lr = models["Lojistik regresyon"]; coefs = pd.Series(lr[-1].coef_[0], index=X.columns).sort_values()
print("\nLojistik regresyon katsayıları (standartlaştırılmış; + hayatta kalmayı artırır):\n", coefs.round(2))
print("\nOdds oranları (1 std. birim artış):\n", np.exp(coefs).round(2))

# Karar ağacı kuralları ve önemleri
from sklearn.tree import export_text
dt = models["Karar ağacı (derinlik 3)"]
print("\nKarar ağacı kuralları:\n", export_text(dt, feature_names=list(X.columns)))
print("Özellik önemleri (derinlik 3):\n", pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).round(3))
res.to_csv(os.path.join(sys.argv[2], "model_sonuclari.csv"))
