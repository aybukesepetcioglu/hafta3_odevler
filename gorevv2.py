import cv2
import numpy as np

resim = cv2.imread('goruntuisleme.jpg', cv2.IMREAD_GRAYSCALE)
yukseklik, genislik = resim.shape
toplam_piksel = yukseklik * genislik

hist = [0] * 256
for i in range(yukseklik):
    for j in range(genislik):
        hist[resim[i, j]] += 1

cdf = [0] * 256
kumulatif_toplam = 0
for k in range(256):
    kumulatif_toplam += hist[k]
    cdf[k] = kumulatif_toplam

cdf_min = next(v for v in cdf if v > 0)

lookup_table = [0] * 256
for k in range(256):
    if cdf[k] == 0:
        lookup_table[k] = 0
    else:
        deger = ((cdf[k] - cdf_min) / (toplam_piksel - cdf_min)) * 255
        lookup_table[k] = int(round(deger))

esitlenmis_resim = np.zeros((yukseklik, genislik), dtype=np.uint8)
for i in range(yukseklik):
    for j in range(genislik):
        esitlenmis_resim[i, j] = lookup_table[resim[i, j]]

cv2.imwrite('odev2_sonuc.jpg', esitlenmis_resim)
cv2.imshow('Odev 2 - Histogram Esitleme', esitlenmis_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()