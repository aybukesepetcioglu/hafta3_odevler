import cv2
import numpy as np

resim = cv2.imread('goruntuisleme.jpg', cv2.IMREAD_GRAYSCALE)

if resim is None:
    raise FileNotFoundError("goruntuisleme.jpg bulunamadı.")

yukseklik, genislik = resim.shape

min_val = 255
max_val = 0

for i in range(yukseklik):
    for j in range(genislik):
        val = int(resim[i, j])
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

print(f"Minimum Değer: {min_val}, Maksimum Değer: {max_val}")

olceklenmis_resim = np.zeros((yukseklik, genislik), dtype=np.uint8)

payda = max_val - min_val if max_val != min_val else 1

for i in range(yukseklik):
    for j in range(genislik):
        piksel = int(resim[i, j])
        yeni_piksel = ((piksel - min_val) / payda) * 255.0
        olceklenmis_resim[i, j] = int(round(yeni_piksel))

cv2.imwrite('odev1_sonuc.jpg', olceklenmis_resim)
cv2.imshow('Orijinal', resim)
cv2.imshow('Odev 1 - Dogrusal Olcekleme', olceklenmis_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()