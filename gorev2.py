import cv2
import numpy as np

resim = cv2.imread("goruntuisleme_2.jpg", cv2.IMREAD_GRAYSCALE)

if resim is None:
    print("Hata: Resim bulunamadı! Dosya adını kontrol edin.")
    exit()

yukseklik, genislik = resim.shape
toplam_piksel = yukseklik * genislik

histogram = [0] * 256

for i in range(yukseklik):
    for j in range(genislik):
        piksel_degeri = resim[i, j]
        histogram[piksel_degeri] += 1

cdf = [0] * 256
birikimli_toplam = 0

for i in range(256):
    birikimli_toplam += histogram[i]
    cdf[i] = birikimli_toplam

cdf_min = 0
for deger in cdf:
    if deger > 0:
        cdf_min = deger
        break

donusum_tablosu = [0] * 256

payda = toplam_piksel - cdf_min

if payda > 0:
    for v in range(256):
        if cdf[v] < cdf_min:
            donusum_tablosu[v] = 0
        else:
            yeni_deger = ((cdf[v] - cdf_min) / payda) * 255
            donusum_tablosu[v] = int(round(np.clip(yeni_deger, 0, 255)))
else:
    for v in range(256):
        donusum_tablosu[v] = v

esitlenmis_resim = np.zeros((yukseklik, genislik), dtype=np.uint8)

for i in range(yukseklik):
    for j in range(genislik):
        eski_deger = resim[i, j]
        esitlenmis_resim[i, j] = donusum_tablosu[eski_deger]

cv2.imwrite("esitlenmis_cikti.jpg", esitlenmis_resim)
print("İşlem tamamlandı! Sonuç 'esitlenmis_cikti.jpg' olarak kaydedildi.")

yan_yana = np.hstack((resim, esitlenmis_resim))
cv2.imshow("Sol: Orijinal Sisli | Sag: Histogram Esitlenmis", yan_yana)

cv2.waitKey(0)
cv2.destroyAllWindows()