# Gaussian Naive Bayes Sınıflandırıcı

Bu projede, **Gaussian Naive Bayes (GNB)** algoritması hem **manuel olarak sıfırdan** hem de **`scikit-learn` kütüphanesi** kullanılarak uygulanmış ve karşılaştırılmıştır.

---

##  Özellikler
- **Manuel GNB Modeli (`model.py`)**  
  - NumPy kullanılarak sıfırdan yazılmıştır.  
  - Her sınıf için:
    - **Prior (öncelik) olasılıkları**,
    - **Özellik başına ortalama ve varyans** hesaplanır.
  - Gaussian PDF formülü manuel olarak uygulanır (mean/var fonksiyonları kullanılmadan).
  - Test doğruluğu: **`0.978`**

- **Hazır GNB Modeli (`sklearn.naive_bayes.GaussianNB`)**  
  - `scikit-learn` kütüphanesi kullanılmıştır.  
  - Test doğruluğu: **`0.9777`**

---

##  Test Dosyaları
| Dosya Adı             | Açıklama                                  |
|-----------------------|--------------------------------------------|
| `manuelmodel_test.py` | Manuel modelin eğitilip test edildiği dosya |
| `hazirmodel_test.py`  | Sklearn modelinin test edildiği dosya       |

---

##  Sonuç
- Manuel olarak geliştirilen model, hazır `sklearn` modeliyle **neredeyse aynı doğruluk oranını** sağlamıştır.
- Bu çalışma, Gaussian Naive Bayes algoritmasının matematiksel temellerinin öğrenilmesi açısından değerli bir uygulamadır.

