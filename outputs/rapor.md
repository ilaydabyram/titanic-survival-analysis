# Titanik: Kim hayatta kaldı? (Proje 1)

## 1. Soru

> Bir Titanik yolcusunun hayatta kalıp kalmadığını yaşı, cinsiyeti ve bilet sınıfı gibi bilgilerinden tahmin edebilir miyiz? Hayatta kalmayı en çok hangi faktörler etkilemiş?

Bu bir **sınıflandırma** problemi: tahmin etmek istediğimiz şey iki değerden birini alıyor (hayatta kaldı = 1, kaldı = 0).

## 2. Veri

- **Kaynak:** Titanik yolcu kayıtları (seaborn-data deposundaki `titanic.csv`), gerçek yolcu verisi.
- **Boyut:** 891 yolcu, 15 sütun.
- **Hedef sütun:** `survived`. Yolcuların %38,4'ü hayatta kalmış.
- **Kullandığımız özellikler:** bilet sınıfı (`pclass`), cinsiyet (`sex`), yaş (`age`), kardeş/eş sayısı (`sibsp`), ebeveyn/çocuk sayısı (`parch`), bilet ücreti (`fare`), bindiği liman (`embarked`).

## 3. Temizlik

Modele vermeden önce veriyi hazırladık. Her kararın nedeni:

| Sorun | Ne yaptık | Neden |
|---|---|---|
| `alive` sütunu `survived` ile aynı bilgiyi metinle veriyor | Attık | Bu "veri sızıntısı": model cevabı doğrudan görürdü, öğrenmiş olmazdı. |
| `who`, `adult_male`, `class`, `embark_town`, `alone` | Attık | Başka sütunlardan türetilmiş, yeni bilgi vermiyor. |
| `deck` sütununun %77'si boş (688 satır) | Attık | Bu kadar eksik veriyi doldurmak uydurmak olur. |
| `age` sütununda 177 eksik (%20) | Aynı sınıf ve cinsiyetteki yolcuların medyan yaşıyla doldurduk | Satırları atsaydık verinin beşte birini kaybederdik. |
| `embarked` sütununda 2 eksik | En sık görülen liman (Southampton) ile doldurduk | Sadece 2 satır, etkisi yok denecek kadar az. |
| 107 "tekrar eden" satır | Dokunmadık | Yolcu numarası yok. Aynı özelliklere sahip farklı yolcular olabilir, tekrar diye atmak doğru olmaz. |

Yeni özellik: `family_size = sibsp + parch + 1` (yolcunun aile büyüklüğü, kendisi dahil).

## 4. Keşif: veri ne söylüyor?

**Cinsiyet en belirleyici.** Kadınların %74,2'si, erkeklerin ise %18,9'u hayatta kalmış.

**Bilet sınıfı ikinci.** Hayatta kalma oranı 1. sınıfta %63,0, 2. sınıfta %47,3, 3. sınıfta %24,2.

**İkisi birlikte** (Grafik 1, `g1_sinif_cinsiyet.png`):

| | Kadın | Erkek |
|---|---|---|
| 1. sınıf | %96,8 | %36,9 |
| 2. sınıf | %92,1 | %15,7 |
| 3. sınıf | %50,0 | %13,5 |

Üçüncü sınıftaki kadınların hayatta kalma oranı (%50), birinci sınıftaki erkeklerinkinden (%37) bile yüksek. Yani cinsiyet sınıftan daha güçlü bir etki göstermiş.

**Yaş** (Grafik 2, `g2_yas.png`, yaşı bilinen 714 yolcu): 0-12 yaş çocukların %58'i hayatta kalmış. 19-35 yaşta oran %38, 60 yaş üstünde %23 (bu grupta yalnızca 22 kişi var, o yüzden kesin yorum yapmak zor).

**Aile büyüklüğü** (Grafik 3, `g3_aile.png`): Yalnız yolculuk edenlerin %30'u, 2-4 kişilik ailelerin %55-72'si hayatta kalmış. 5 ve üzeri kişilik ailelerde oran %16'ya düşüyor.

**Liman:** Cherbourg'dan binenlerin hayatta kalma oranı %55,4, Southampton'dan binenlerinki %33,9. Cherbourg yolcularının %51'i 1. sınıftaydı, Southampton'da bu oran %20. Sınıf farkın bir kısmını açıklıyor ama hepsini değil: aynı sınıf içinde bile Cherbourg önde (1. sınıfta %69'a karşı %59, 3. sınıfta %38'e karşı %19). Yine de modelde limanın katkısı küçük çıktı, bu yüzden ana sonuçlara dahil etmedik.

## 5. Modelleme

**Yöntem:** Veriyi %80 eğitim (712 yolcu), %20 test (179 yolcu) olarak böldük. Model eğitim kısmını görüyor, testi hiç görmeden ona sonra not veriyoruz. Bölme sırasında hayatta kalma oranı iki tarafta da aynı kalsın diye `stratify` kullandık.

**Neden bir baz çizgi?** "Herkes öldü" diyen aptal bir model bile %61,5 doğru çıkıyor, çünkü yolcuların çoğu hayatta kalamamış. Gerçek modelin işe yaradığını söylemek için bu rakamı geçmesi lazım.

| Model | Eğitim doğruluğu | Çapraz doğrulama (eğitim) | **Test doğruluğu** |
|---|---|---|---|
| Baz çizgi (hep "öldü") | %61,7 | %61,7 | %61,5 |
| Lojistik regresyon | %80,9 | %80,3 | **%81,0** |
| Karar ağacı (derinlik 3) | %82,9 | %80,6 | %80,4 |
| Karar ağacı (derinlik 8) | %91,6 | %80,6 | %78,8 |

**Ne öğreniyoruz bu tablodan?**

1. İki basit model de baz çizgiyi yaklaşık 19 puan geçiyor. Özellikler gerçekten bilgi taşıyor.
2. **Aşırı öğrenme (overfitting):** Derinlik 8'lik ağaç eğitimde %91,6 yapıyor ama testte %78,8'e düşüyor. Ezberlemiş, genellememiş. Derinlik 3'lük ağaç ve lojistik regresyon bu açıdan daha sağlıklı.
3. **Lojistik regresyon ile derinlik 3'lük ağacı kıyaslarken dikkat:** test seti 179 kişi, yani 1 kişi yaklaşık %0,56 eder. %81,0 ile %80,4 arasındaki fark tek bir yolcu. İkisi arasında "biri daha iyi" demek için kanıtımız yok.

**Lojistik regresyonun ayrıntısı** (test seti): Hayatta kalacak dediği yolcuların %78,7'si gerçekten hayatta kalmış (kesinlik). Gerçekte hayatta kalanların %69,6'sını yakalayabilmiş (duyarlılık). Yani hayatta kalan 69 kişiden 21'ini kaçırmış.

## 6. Hayatta kalmayı en çok ne etkilemiş?

Soruya iki modelden aynı cevap geliyor:

- **Karar ağacı (derinlik 3) özellik önemleri:** cinsiyet %63, bilet sınıfı %22, yaş %9, aile büyüklüğü %6. Ücretin ve limanın katkısı yok denecek kadar az.
- **Lojistik regresyon (standartlaştırılmış katsayılar):** en güçlü etki erkek olmak (olasılığı düşürüyor), sonra bilet sınıfı (sınıf numarası büyüdükçe düşüyor), sonra yaş (yaşlandıkça düşüyor), sonra aile büyüklüğü.
- Ağacın kuralları da bunu gösteriyor: kadın ve 1.-2. sınıf ise hayatta kalmış tahmini, erkek ve 3,5 yaşından büyükse hayatta kalamamış tahmini.

Bu, "önce kadınlar ve çocuklar" kuralının verideki izi gibi duruyor.

## 7. Sınırlar (dürüst kısım)

- **Bu bir ilişki, sebep değil.** Veri "cinsiyet hayatta kalmayı belirledi" demiyor. Hangi yolcunun hangi filikaya binebildiğini bilmiyoruz, sadece kimin hayatta kaldığını biliyoruz.
- **Yaşların %20'sini biz doldurduk.** Yaş etkisini yorumlarken bunu aklımızda tutmalıyız.
- **Küçük test seti.** 179 yolcuyla ölçülen doğruluk birkaç puan oynayabilir.
- **Tek bir bölme.** Rakamlar `random_state=42` ile yapılan tek bir eğitim/test bölmesine ait. Çapraz doğrulama sonuçları (yaklaşık %80) bu yüzden daha güvenilir bir ipucu.

## 8. Dosyalar

- `scripts/01_inspect.py`: veriye ilk bakış
- `scripts/02_clean_eda.py`: temizlik, tablolar ve grafikler
- `scripts/03_model.py`: modeller ve değerlendirme
- `outputs/`: temiz veri, grafikler, model sonuçları

Çalıştırma sırası: `01` -> `02` -> `03` (her script veri yolunu argüman olarak alır).

## 9. Sıradaki adım

Bir sonraki projede aynı iskeleti (soru, veri, temizlik, keşif, model, yorum) farklı bir veriyle tekrarlayacağız. Burada hep sınıflandırma yaptık. İstersen sıradaki proje regresyon olsun, yani bir sayıyı tahmin edelim.
