import cv2
import numpy as np

resim = cv2.imread("goruntuisleme_3.jpg", cv2.IMREAD_GRAYSCALE)

if resim is None:
    print("Hata: Resim bulunamadı! Dosya adını kontrol edin.")
    exit()

clahe_hazir = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
resim_opencv_clahe = clahe_hazir.apply(resim)

def manuel_clahe(img, grid_size=(8, 8), clip_limit=2.0):
    h, w = img.shape
    grid_r, grid_c = grid_size

    tile_h = h // grid_r
    tile_w = w // grid_c

    limit = int(clip_limit * (tile_h * tile_w) / 256)
    if limit < 1:
        limit = 1

    lut_tablolari = np.zeros((grid_r, grid_c, 256), dtype=np.uint8)

    for r in range(grid_r):
        for c in range(grid_c):
            y_bas = r * tile_h
            y_bit = (r + 1) * tile_h if r < grid_r - 1 else h
            x_bas = c * tile_w
            x_bit = (c + 1) * tile_w if c < grid_c - 1 else w

            blok = img[y_bas:y_bit, x_bas:x_bit]
            b_h, b_w = blok.shape
            toplam_blok_piksel = b_h * b_w

            
            hist = [0] * 256
            for y in range(b_h):
                for x in range(b_w):
                    hist[blok[y, x]] += 1

            
            fazlalik = 0
            for k in range(256):
                if hist[k] > limit:
                    fazlalik += hist[k] - limit
                    hist[k] = limit

            eklenecek = fazlalik // 256
            kalan = fazlalik % 256

            for k in range(256):
                hist[k] += eklenecek
            for k in range(kalan):
                hist[k] += 1

            cdf = [0] * 256
            toplam = 0
            for k in range(256):
                toplam += hist[k]
                cdf[k] = toplam

            for k in range(256):
                lut_tablolari[r, c, k] = int(
                    round((cdf[k] / toplam_blok_piksel) * 255)
                )
    cikti = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            r = (y - tile_h // 2) / tile_h
            c = (x - tile_w // 2) / tile_w

            r1 = int(np.floor(r))
            c1 = int(np.floor(c))
            r2 = r1 + 1
            c2 = c1 + 1
            r1 = np.clip(r1, 0, grid_r - 1)
            r2 = np.clip(r2, 0, grid_r - 1)
            c1 = np.clip(c1, 0, grid_c - 1)
            c2 = np.clip(c2, 0, grid_c - 1)

            
            delta_r = r - np.floor(r)
            delta_c = c - np.floor(c)

            val = img[y, x]

            v11 = lut_tablolari[r1, c1, val]
            v12 = lut_tablolari[r1, c2, val]
            v21 = lut_tablolari[r2, c1, val]
            v22 = lut_tablolari[r2, c2, val]

            ust = (1 - delta_c) * v11 + delta_c * v12
            alt = (1 - delta_c) * v21 + delta_c * v22
            yeni_val = (1 - delta_r) * ust + delta_r * alt

            cikti[y, x] = int(round(np.clip(yeni_val, 0, 255)))

    return cikti


resim_manuel_clahe = manuel_clahe(resim, grid_size=(8, 8), clip_limit=2.0)

cv2.imwrite("clahe_opencv.jpg", resim_opencv_clahe)
cv2.imwrite("clahe_manuel.jpg", resim_manuel_clahe)

karsilastirma = np.hstack((resim, resim_opencv_clahe, resim_manuel_clahe))
cv2.imshow("Sol: Orijinal | Orta: OpenCV CLAHE | Sag: Manuel CLAHE", karsilastirma)

cv2.waitKey(0)
cv2.destroyAllWindows()