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

# Convert dari BGR ke RGB untuk visualisasi citra asli yang berwarna
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Convert citra asli ke grayscale untuk proses deteksi tepi
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

print("Ukuran citra (berwarna):", img_color.shape)
print("Ukuran citra (grayscale):", img_gray.shape)
print("Tipe data citra:", img_gray.dtype)

# ==========================================
# OPERASI DETEKSI TEPI PREWITT (Tanpa Noise)
# ==========================================
kernel_prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

kernel_prewitt_y = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# Filter 2D langsung diterapkan ke citra img_gray
prewitt_x_float = cv2.filter2D(img_gray, cv2.CV_32F, kernel_prewitt_x)
prewitt_x = cv2.convertScaleAbs(prewitt_x_float)

prewitt_y_float = cv2.filter2D(img_gray, cv2.CV_32F, kernel_prewitt_y)
prewitt_y = cv2.convertScaleAbs(prewitt_y_float)

prewitt_mag_float = cv2.magnitude(prewitt_x_float, prewitt_y_float)

# Normalisasi untuk visualisasi
prewitt_mag = cv2.normalize(prewitt_mag_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Simpan hasil ke folder outputs
cv2.imwrite("outputs/prewitt_x.png", prewitt_x)
cv2.imwrite("outputs/prewitt_y.png", prewitt_y)
cv2.imwrite("outputs/prewitt_magnitude.png", prewitt_mag)

# ==========================================
# MENAMPILKAN HASIL
# ==========================================
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli (Berwarna)")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(prewitt_x, cmap='gray', vmin=0, vmax=255)
plt.title("Prewitt X")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(prewitt_y, cmap='gray', vmin=0, vmax=255)
plt.title("Prewitt Y")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(prewitt_mag, cmap='gray', vmin=0, vmax=255)
plt.title("Prewitt Magnitude")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/hasil_perbandingan_prewitt_tanpa_noise.png", bbox_inches='tight')
plt.show()