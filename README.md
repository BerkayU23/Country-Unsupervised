   # Ülkelerin Sosyo-Ekonomik Gelişmişlik Düzeylerine Göre Kümelenmesi (Unsupervised Learning)

Bu proje, ülkelerin temel ekonomik ve sağlık göstergelerini analiz ederek, benzer özelliklere sahip ülkeleri gruplandırmak amacıyla geliştirilmiştir. Proje, özellikle yardım kuruluşları ve strateji uzmanları için bütçe önceliklendirmesi yaparken veriye dayalı bir temel sunmayı hedefler.

## 📁 Proje Yapısı

Proje içerisinde iki temel çalışma dosyası bulunmaktadır:

*   **`Country-Unsupervised.ipynb` (Jupyter Notebook):** Projenin görsel analiz raporudur. Adım adım kodların çalıştırılmış hallerini, veri tablolarını ve interaktif harita sonuçlarını içerir. **Doğrudan inceleme için bu dosya önerilir.**
*   **`Country Unsupervised.py` (Python Script):** Projenin otomatize edilebilir saf kod versiyonudur.

## 🚀 Uygulanan Analiz Adımları

1.  **Keşifçi Veri Analizi (EDA):** Korelasyon ısı haritaları ve histogramlar ile veri setindeki yapısal ilişkiler incelendi.
2.  **Veri Ön İşleme:** `MinMaxScaler` ile normalizasyon yapılarak tüm değişkenler aynı ölçeğe getirildi. Ülke isimleri gibi kategorik veriler, model doğruluğu için işlem dışı tutuldu.
3.  **Boyut İndirgeme (PCA):** Verinin gürültüsünü azaltmak ve performansı artırmak için **Principal Component Analysis** kullanılarak boyut 3 temel bileşene düşürüldü.
4.  **Kümeleme Modelleri:** 
    *   **K-Means:** `KneeLocator` ile ideal küme sayısı (dirsek noktası) belirlenerek en dengeli sonuçlar elde edildi.
    *   **Hierarchical Clustering (Agglomerative):** Hiyerarşik yapı incelendi.
    *   **DBSCAN & HDBSCAN:** Yoğunluk tabanlı kümeleme denemeleri yapıldı ve hiperparametre optimizasyonu (`eps`, `min_samples`) ile en yüksek Silhouette skorları hedeflendi.
5.  **Görselleştirme:** Kümeleme sonuçları "Bütçe Gereken", "Arada Kalan" ve "Bütçe Gerekmeyen" şeklinde etiketlenerek **Plotly** üzerinden interaktif bir dünya haritasına (map.png) aktarıldı.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

* **Dil:** Python
* **Veri Manipülasyonu:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-learn (PCA, K-Means, Agglomerative, DBSCAN), HDBSCAN
* **Yardımcı Araçlar:** kneed (Elbow noktası tespiti)
* **Görselleştirme:** Matplotlib, Seaborn, Plotly Express

## 📊 Önemli Bulgular ve Sonuçlar

*   **PCA Etkisi:** Boyut indirgeme işlemi, K-Means ve HC gibi mesafe tabanlı algoritmaların Silhouette skorlarını belirgin şekilde iyileştirmiştir.
*   **Algoritma Karşılaştırması:** DBSCAN yoğunluk farklarından dolayı tek bir dev küme üretme eğilimi gösterirken, **K-Means (k=3)** veriyi ekonomik gerçekliklerle en uyumlu şekilde gruplandıran model olmuştur (Silhouette Score: ~0.438).

## 🛠️ Kurulum

```bash
git clone [https://github.com/BerkayU23/Country-Unsupervised.git](https://github.com/BerkayU23/Country-Unsupervised.git)
cd Country-Unsupervised
pip install pandas numpy scikit-learn seaborn matplotlib plotly kneed hdbscan
