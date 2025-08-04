import numpy as np
import math
from collections import defaultdict


class GNaiveBayes:
    def __init__(self):
        self.etiketler=None
        self.ozellikler=None
        self.prior=None
        self.ort=None
        self.varyans=None
        self.alt=None
        self.etiket_sayilari=None
        self.satir_indeksleri=None  
        self.alt=None   
    
    def fit(self, X, y):
        "X ve y icin model egitilmeli "
         
        X = np.array(X)  
        y = np.array(y)     
        self.X= X  
        self.y= y    
        "siniflarin priorsini ve her sinif icin ortalama ve varyans degerlerini hesaplamali "
        #ortalamam hesabi icin toplam örnek sayisi ve kac cesit etiket oldugunu bilmeli
        #priors P(setosa)= setosa ile etiketlenen sayisi/örnek sayisi
        #örnek sayisi = X'in satir sayisi
        
        örnek_sayisi= X.shape[0] 
        self.etiketler, self.etiket_sayilari = np.unique(y, return_counts=True)  #etiket ceistleri ve sayilari
        self.prior = {}
        for i, count in zip(self.etiketler, self.etiket_sayilari): 
           olasilik = count / örnek_sayisi
           self.prior[i] = olasilik  # zip iki listeyi eslestirir ("spam", 12) gibi



        "her sinif ve özellik icin ortalama hesaplanmali "
        #setosa iki sinifta etiketli ise özellik1 icin ort  (örnek1+örnek2)/2 gibi bir deger bu her özellik icin ayri ayri hesaplanmali
        #özellik sayisina gerek yok döngüyle hepsine girsin
        self.etiketler = self.etiketler.tolist()  #etiketleri listeye cevirir ayni sartorda bulunan etiketleri bulursam uzerinden islem yapabilirim
        satir_indeksleri= defaultdict(list) #etiketlere göre satir indekslerini grupla

        for i, etiket in enumerate(self.y): #etiketlerin satir indekslerini bulur
            satir_indeksleri[etiket].append(i) # etiketlere göre satir indekslerini gruplar
        ozellikler = X.shape[1]
        self.ozellikler = list(range(ozellikler))

        self.ort = {}
        for etiket in self.etiketler:
            alt = np.array(satir_indeksleri[etiket], dtype=int)  
            
            for j in  self.ozellikler:
                degerler = self.X[alt, [j]].tolist()
                toplam = 0
                for x in degerler:
                    toplam += x
                ort = toplam / len(degerler)
                self.ort[(etiket, j)] = ort
                
                #self.ort[(etiket, j)] = degerler.mean()  # her özellik icin ortalama deger
           #ort= toplam / adet
           #self.ort[oz] = ort  # her özellik icin ortalama deger

        "her sinif ve özellik icin varyans hesaplanmali "
        #setosa iki sinifta etiketli ise özellik1 icin varyans hesaplanmali  varyans = ((örnek1-ort)^2 + (örnek2-ort)^2)/2gibi bir deger bu her özellik icin ayri ayri hesaplanmali
        self.varyans = {}
        for etiket in self.etiketler:
            alt = np.array(satir_indeksleri[etiket], dtype=int)  #etikete göre satir indekslerini alir
            for j in self.ozellikler:
                degerler= self.X[alt, j].tolist()
                ort = self.ort[(etiket, j)]
                kare_farklar_toplam = 0
                for x in degerler:
                    fark = x - ort 
                    kare = fark ** 2
                    kare_farklar_toplam += kare
                varyans = kare_farklar_toplam / len(degerler) 
                self.varyans[(etiket, j)] = varyans  
               # self.varyans[(etiket, j)] = degerler.var()  # her özellik icin varyans degerini hesaplar
            """ for i in satir_indeksleri[self.etiketler[0]]:
                varyans += (df[oz][i] - self.ort[oz]) ** 2 """
            

    def predict(self, X):
      "X icin etiket tahminleri yapmali "
    
      "her sinif icin prior degeri  özelliklerin ortalamasi ve varyansi kullanilarak PDF hesaplanir"
      # 1/(sqrt(2pi * varyans(mesela sepal_length icin setosada hesaplanan varyans degeri))) * exp(-((x(test icin gelen sepal_length degeri)- ort(sepal_length icin setosada hesaplanan ort degeri))^2) / (2 * varyans(sepal_length icin setosada hesaplanan varyans degeri)))

      X = np.array(X)  
      y_test = []
      for xi in X:
          olasiliklar = {}
          for etiket in self.etiketler:   
              toplam_log = math.log(self.prior[etiket])  
              for j in range(X.shape[1]):
                  ort = self.ort[(etiket, j)]
                  varyans = self.varyans[(etiket, j)]
                  if varyans <= 0:
                      varyans = 1e-9  # varyans sifir veya negatif ise hata vermemesi icin 

                  pdf = 1 / (math.sqrt(2 * math.pi * varyans)) * math.exp(-((xi[j] - ort) ** 2) / (2 * varyans))
                  if pdf <= 0:
                   pdf = 1e-10 # PDF sifir veya negatif ise hata vermemesi icin
                  toplam_log += math.log(pdf)
              olasiliklar[etiket] = toplam_log

          #log olasilik hesaplamalari yapilir log(prior) ->baslangicta bulunan deger+ log(PDF1) + log(PDF2) + ... + log(PDFn) -> setosa icin her özellikte ayri ayri hesaplanan PDF degerleri
          #bu toplamdan elde edilen deger P(class|X) olasiligini verir yani X'in setosa olma olasiligi 
          #her sinif icin bu islemler yapilir ve en yuksek olasiliga sahip sinif secilir
          "sonuc olarak en yuksek olasiliga sahip sinifin etiketini dondur"
          tahmin = max(olasiliklar, key=olasiliklar.get)
          y_test.append(tahmin)

      return y_test
