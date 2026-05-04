# Ülkelerin Sosyo-Ekonomik Gelişmişlik Düzeylerine Göre Kümelenmesi (Unsupervised Learning)

Bu proje, ülkelerin sosyo-ekonomik ve sağlık göstergelerini (çocuk ölüm oranları, ihracat, sağlık harcamaları, ithalat, gelir, enflasyon, yaşam beklentisi, doğurganlık oranı ve GSYİH) kullanarak ülkeleri benzerliklerine göre gruplandırmayı amaçlayan bir **Gözetimsiz Öğrenme (Unsupervised Learning)** çalışmasıdır.

Temel hedef, ülkelere yapılabilecek olası finansal yardımları veya bütçe desteklerini önceliklendirmek için veriye dayalı bir kümeleme (Clustering) yapmaktır.

## 🚀 Proje Adımları ve Metodoloji

1. **Keşifçi Veri Analizi (EDA):** Veri setindeki özelliklerin dağılımlarını anlamak için histogramlar ve özellikler arası ilişkileri incelemek için korelasyon ısı haritaları (heatmap) oluşturuldu.
2. **Veri Ön İşleme (Data Preprocessing):**
   * Metin verileri (Ülke isimleri) model eğitiminden önce veri sızıntısını (data leakage) önlemek adına dikkatlice ayrıştırıldı.
   * Özelliklerin farklı ölçeklerde olmasının mesafeye dayalı algoritmaları yanıltmaması için `MinMaxScaler` ile normalizasyon uygulandı.
3. **Boyut İndirgeme (Dimensionality Reduction):** Veri setinin taşıdığı varyansın %90'ından fazlasını açıklayan ilk 3 bileşen, **PCA (Principal Component Analysis)** kullanılarak elde edildi ve modellerin daha gürbüz (robust) çalışması sağlandı.
4. **Modelleme:** Boyutları indirgenmiş ve normalleştirilmiş veri seti üzerinde aşağıdaki kümeleme algoritmaları eğitildi:
   * **K-Means** (İdeal küme sayısı `KneeLocator` ile dirsek yöntemi kullanılarak belirlendi)
   * **Agglomerative Hierarchical Clustering (HC)**
   * **DBSCAN**
   * **HDBSCAN**
5. **Hiperparametre Optimizasyonu:** Yoğunluk tabanlı algoritmaların (DBSCAN ve HDBSCAN) tek bir dev küme üretmesini (underfitting) engellemek için `eps`, `min_samples` ve `min_cluster_size` gibi parametreler üzerinde döngüler kurularak detaylı Grid Search benzeri optimizasyonlar yapıldı.
6. **Değerlendirme:** Modellerin başarısı **Silhouette Score** metrikleri üzerinden PCA'li ve PCA'siz durumlar için karşılaştırmalı olarak analiz edildi.
7. **Görselleştirme:** Elde edilen en tutarlı etiketler ("Bütçe Gerekli", "Arada Kalanlar", "Bütçe Gerekli Değil"), **Plotly** kullanılarak dünya haritası (Choropleth Map) üzerinde coğrafi olarak görselleştirildi.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

* **Dil:** Python
* **Veri Manipülasyonu:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-learn (PCA, K-Means, Agglomerative, DBSCAN), HDBSCAN
* **Yardımcı Araçlar:** kneed (Elbow noktası tespiti)
* **Görselleştirme:** Matplotlib, Seaborn, Plotly Express

## 📊 Öne Çıkan Sonuçlar

* PCA uygulamanın, bu veri seti özelinde tüm kümeleme algoritmalarının Silhouette skorlarını belirgin şekilde artırdığı gözlemlenmiştir.
* Yoğunluk tabanlı algoritmalar (DBSCAN/HDBSCAN) veri setinin yapısı gereği tek bir büyük küme eğilimi gösterirken, merkez tabanlı **K-Means algoritması** veriyi 3 mantıklı sınıfa bölerek (Silhouette Score: ~0.438) ekonomik gerçekliklerle örtüşen en dengeli sonuçları üretmiştir.

## 💻 Kurulum ve Kullanım

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/country-clustering.git](https://github.com/KULLANICI_ADINIZ/country-clustering.git)

   pip install pandas numpy scikit-learn seaborn matplotlib plotly kneed hdbscan
   
