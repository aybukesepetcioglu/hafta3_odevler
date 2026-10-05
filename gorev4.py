import cv2
import numpy as np

resim = cv2.imread("goruntuisleme_3.jpg", cv2.IMREAD_GRAYSCALE)
h, w = resim.shape

filtrelenmis_resim = np.zeros((h, w), dtype=np.uint8)

for i in range(1, h - 1):
    for j in range(1, w - 1):
     
        toplam = 0
        for ki in range(-1, 2):
            for kj in range(-1, 2):
                toplam += int(resim[i + ki, j + kj])

        ortalama = toplam // 9
        filtrelenmis_resim[i, j] = ortalama

cv2.imshow("Gurultulu Orijinal", resim)
cv2.imshow("3x3 Ortalama Filtresi Sonucu", filtrelenmis_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()