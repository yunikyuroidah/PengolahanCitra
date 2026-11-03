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
# OPERASI DETEKSI TEPI CANNY (Tanpa Noise)
# ==========================================

# 1. Canny Rendah: Threshold (10, 50) 
# Mengesan banyak butiran tepi, termasuk tepi yang lemah
canny_rendah = cv2.Canny(img_gray, 10, 50)

# 2. Canny Sedang: Threshold (50, 150)
# Nilai standard yang seimbang untuk kebanyakan imej
canny_sedang = cv2.Canny(img_gray, 50, 150)

# 3. Canny Tinggi: Threshold (150, 250)
# Hanya mengesan tepi yang sangat kuat dan jelas (garis utama)
canny_tinggi = cv2.Canny(img_gray, 150, 250)

# Simpan hasil ke folder outputs
cv2.imwrite("outputs/canny_rendah.png", canny_rendah)
cv2.imwrite("outputs/canny_sedang.png", canny_sedang)
cv2.imwrite("outputs/canny_tinggi.png", canny_tinggi)

# ==========================================
# MENAMPILKAN HASIL
# ==========================================
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli (Berwarna)")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(canny_rendah, cmap='gray', vmin=0, vmax=255)
plt.title("Canny Rendah (30, 80)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(canny_sedang, cmap='gray', vmin=0, vmax=255)
plt.title("Canny Sedang (100, 200)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(canny_tinggi, cmap='gray', vmin=0, vmax=255)
plt.title("Canny Tinggi (150, 250)")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/hasil_perbandingan_canny_tanpa_noise.png", bbox_inches='tight')
plt.show()