import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True, parents=True)

# Membaca gambar asli berwarna
img_color = cv2.imread("assets/images1.jpg")

if img_color is None:
    raise FileNotFoundError(
        "Citra images1.jpg tidak ditemukan. "
        "Pastikan file berada di folder yang sama"
    )

# Matplotlib menggunakan format RGB, sedangkan OpenCV menggunakan BGR
# Jadi kita perlu convert dari BGR ke RGB untuk visualisasi citra asli
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Convert citra asli ke grayscale untuk proses penambahan noise dan deteksi tepi
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Menambahkan Noise pada Gambar Grayscale
noisy = img_gray.copy()
rng = np.random.default_rng(42)
noise_mask = rng.random(img_gray.shape[:2])
noisy[noise_mask < 0.02] = 0
noisy[noise_mask > 0.98] = 255

print("Ukuran citra (berwarna):", img_color.shape)
print("Ukuran citra (grayscale/noisy):", noisy.shape)
print("Tipe data citra:", noisy.dtype)

# ==========================================
# OPERASI DETEKSI TEPI SOBEL
# ==========================================

# 1. Sobel X (Deteksi Tepi Vertikal)
sobel_x_float = cv2.Sobel(noisy, cv2.CV_32F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x_float)

# 2. Sobel Y (Deteksi Tepi Horizontal)
sobel_y_float = cv2.Sobel(noisy, cv2.CV_32F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y_float)

# 3. Sobel Magnitude (Gabungan X dan Y)
sobel_mag_float = cv2.magnitude(sobel_x_float, sobel_y_float)
# Normalisasi untuk visualisasi
sobel_mag = cv2.normalize(sobel_mag_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Simpan hasil ke folder outputs
cv2.imwrite("outputs/noisy_sobel.png", noisy)
cv2.imwrite("outputs/sobel_x.png", sobel_x)
cv2.imwrite("outputs/sobel_y.png", sobel_y)
cv2.imwrite("outputs/sobel_magnitude.png", sobel_mag)

# Menampilkan hasil
plt.figure(figsize=(15, 4))

plt.subplot(1, 5, 1)
# Menampilkan citra berwarna (RGB)
plt.imshow(img_rgb)
plt.title("Citra Asli (Berwarna)")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(noisy, cmap='gray', vmin=0, vmax=255)
plt.title("Citra Noisy")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(sobel_x, cmap='gray', vmin=0, vmax=255)
plt.title("Sobel X")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(sobel_y, cmap='gray', vmin=0, vmax=255)
plt.title("Sobel Y")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(sobel_mag, cmap='gray', vmin=0, vmax=255)
plt.title("Sobel Magnitude")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/hasil_perbandingan_sobel.png", bbox_inches='tight')
plt.show()