import cv2
import numpy as np

resim = cv2.imread('goruntuisleme.jpg', cv2.IMREAD_GRAYSCALE)

clahe_cv = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
resim_clahe_cv = clahe_cv.apply(resim)
cv2.imwrite('odev3_clahe_opencv.jpg', resim_clahe_cv)

def manuel_clahe(img, grid_size=(8, 8), clip_limit=2.0):
    H, W = img.shape
    num_grid_y, num_grid_x = grid_size
    tile_h = H // num_grid_y
    tile_w = W // num_grid_x

    luts = np.zeros((num_grid_y, num_grid_x, 256), dtype=np.float32)
    
    for gy in range(num_grid_y):
        for gx in range(num_grid_x):
            y_start = gy * tile_h
            y_end = (gy + 1) * tile_h if gy != num_grid_y - 1 else H
            x_start = gx * tile_w
            x_end = (gx + 1) * tile_w if gx != num_grid_x - 1 else W
            
            tile = img[y_start:y_end, x_start:x_end]
            tile_pixels = tile.size
         
            hist = np.bincount(tile.ravel(), minlength=256)
           
            actual_clip = int(clip_limit * (tile_pixels / 256.0))
            excess = 0
            for k in range(256):
                if hist[k] > actual_clip:
                    excess += (hist[k] - actual_clip)
                    hist[k] = actual_clip
            
            bonus = excess // 256
            hist += bonus
         
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0][0] if len(cdf[cdf > 0]) > 0 else 0
            denom = tile_pixels - cdf_min if tile_pixels != cdf_min else 1
            lut = np.round(((cdf - cdf_min) / denom) * 255.0)
            luts[gy, gx] = np.clip(lut, 0, 255)
            
    output = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            gy = min((i - tile_h // 2) / tile_h, num_grid_y - 1.0001)
            gx = min((j - tile_w // 2) / tile_w, num_grid_x - 1.0001)
            
            gy1 = int(np.floor(gy))
            gx1 = int(np.floor(gx))
            gy2 = min(gy1 + 1, num_grid_y - 1)
            gx2 = min(gx1 + 1, num_grid_x - 1)
            
            gy1 = max(gy1, 0)
            gx1 = max(gx1, 0)
            
            ay = gy - np.floor(gy) if gy >= 0 else 0
            ax = gx - np.floor(gx) if gx >= 0 else 0
            
            val = img[i, j]
            res_val = ((1 - ax) * (1 - ay) * luts[gy1, gx1, val] +
                       ax * (1 - ay) * luts[gy1, gx2, val] +
                       (1 - ax) * ay * luts[gy2, gx1, val] +
                       ax * ay * luts[gy2, gx2, val])
            output[i, j] = int(round(res_val))
            
    return output

resim_clahe_manuel = manuel_clahe(resim)
cv2.imwrite('odev3_clahe_manuel.jpg', resim_clahe_manuel)

cv2.imshow('CLAHE OpenCV', resim_clahe_cv)
cv2.imshow('CLAHE Manuel', resim_clahe_manuel)
cv2.waitKey(0)
cv2.destroyAllWindows()