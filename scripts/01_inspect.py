import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
print("Şekil:", df.shape)
print("\nDtype ve eksik sayısı:")
print(pd.DataFrame({"dtype": df.dtypes.astype(str), "eksik": df.isna().sum(), "eksik_%": (df.isna().mean()*100).round(1), "benzersiz": df.nunique()}))
print("\nHayatta kalma oranı:", round(df["survived"].mean(), 3))
print("\nSayısal özet:")
print(df[["age","fare","sibsp","parch"]].describe().round(2))
print("\nTekrar eden satır:", df.duplicated().sum())
