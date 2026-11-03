import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
Path("outputs").mkdir(exist_ok=True, parents=True)

img = cv2.imread("assets/images1.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        "Citra images1.jpg tidak ditemukan"
        "Pastikan file berada di folder yang sama"
    )

#Menambahkan Noise pada Gambar
noisy = img.copy()
rng = np.random.default_rng(42)
noise_mask = rng.random(img.shape[:2])
noisy[noise_mask < 0.02] = (0)
noisy[noise_mask > 0.98] = (255)

print("Ukuran citra:", noisy.shape)
print("Tipe data citra:", noisy.dtype)

#HPF 1 : kernel high-pass / deteksi tepi (Laplacian)
hpf1 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32)

#HPF 2 : kernel sharpening / high-boost
hpf2 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

#HPF 3 : kernel high-pass
hpf3 = np.array([
    [1, -2, 1],
    [-2, 5, -2],
    [1, -2, 1]
], dtype=np.float32)

#Menerapkan hpf
J1_float = cv2.filter2D(
    noisy, cv2.CV_32F, hpf1,
    borderType=cv2.BORDER_CONSTANT
)
J2_float = cv2.filter2D(
    noisy, cv2.CV_32F, hpf2,
    borderType=cv2.BORDER_CONSTANT
)
J3_float = cv2.filter2D(
    noisy, cv2.CV_32F, hpf3,
    borderType=cv2.BORDER_CONSTANT
)

J1 = np.clip(np.rint(J1_float), 0, 255).astype(np.uint8)
J2 = np.clip(np.rint(J2_float), 0, 255).astype(np.uint8)
J3 = np.clip(np.rint(J3_float), 0, 255).astype(np.uint8)

# Simpan hasil ke folder outputs
cv2.imwrite("outputs/noisy_hpf.png", noisy)
cv2.imwrite("outputs/hpf1.png", J1)
cv2.imwrite("outputs/hpf2.png", J2)
cv2.imwrite("outputs/hpf3.png", J3)

#Menampilkan hasil
plt.figure(figsize=(15, 4))

plt.subplot(1, 5, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(noisy, cmap='gray', vmin=0, vmax=255)
plt.title("Citra Noisy")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(J1, cmap='gray', vmin=0, vmax=255)
plt.title("Hasil HPF1")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(J2, cmap='gray', vmin=0, vmax=255)
plt.title("Hasil HPF2")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(J3, cmap='gray', vmin=0, vmax=255)
plt.title("Hasil HPF3")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/hasil_perbandingan_hpf.png", bbox_inches='tight')
plt.show()

