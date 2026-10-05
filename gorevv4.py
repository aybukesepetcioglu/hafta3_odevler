import cv2
import numpy as np

resim = cv2.imread('goruntuisleme.jpg', cv2.IMREAD_GRAYSCALE)
H, W = resim.shape

padded = np.zeros((H + 2, W + 2), dtype=np.float32)
padded[1:H+1, 1:W+1] = resim

sonuc_resim = np.zeros((H, W), dtype=np.uint8)

for i in range(H):
    for j in range(W):
        bolge = padded[i:i+3, j:j+3]
        toplam = 0.0
        for r in range(3):
            for c in range(3):
                toplam += bolge[r, c]
      
        sonuc_resim[i, j] = int(round(toplam / 9.0))

cv2.imwrite('odev4_sonuc.jpg', sonuc_resim)
cv2.imshow('Orijinal', resim)
cv2.imshow('Odev 4 - 3x3 Ortalama Filtresi', sonuc_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()