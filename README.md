# Praktikum Pengolahan Citra Digital

> **Disclaimer / Catatan:**
> Repositori ini dibuat khusus untuk keperluan penyelesaian **tugas kuliah** dan sebagai sarana **pembelajaran** materi mata kuliah Pengolahan Citra Digital (Digital Image Processing).

---

## 📌 Deskripsi Proyek

Repositori ini memuat implementasi beberapa teknik dasar pemrosesan citra digital (*Digital Image Processing*) berbasis Python menggunakan library OpenCV, NumPy, dan Matplotlib. Di dalamnya mencakup penambahan gangguan (*noise*), pemfilteran spasial (*Spatial Filtering*), serta berbagai metode deteksi tepi (*Edge Detection*).

---

## 📂 Struktur Berkas

```text
PengolahanCitra/
├── assets/                  # Citra masukan (gambar uji)
│   ├── images1.jpg
│   ├── images2.jpg
│   └── image3.png
├── outputs/                 # Hasil keluaran pemrosesan & visualisasi
├── canny.py                 # Deteksi tepi menggunakan metode Canny
├── hpf8.py                  # High Pass Filter (HPF) untuk penajaman citra
├── ipf8.py                  # Low Pass Filter (LPF) untuk penghalusan citra
├── prewitt.py               # Deteksi tepi menggunakan operator Prewitt
├── roberts.py               # Deteksi tepi menggunakan operator Roberts Cross
├── sobel.py                 # Deteksi tepi menggunakan operator Sobel
├── requirements.txt         # Daftar pustaka / dependensi yang dibutuhkan
└── README.md                # Dokumentasi proyek
```

---

## 🛠️ Materi & Metode yang Dipelajari

1. **Filtering Spasial (Spatial Filtering)**
   - **Low Pass Filter (LPF)**: Digunakan untuk mereduksi derau (*noise*) dan memperhalus citra (*smoothing / blurring*).
   - **High Pass Filter (HPF)**: Digunakan untuk menonjolkan detail dan mempertajam tepi citra (*sharpening*).

2. **Deteksi Tepi (Edge Detection)**
   - **Sobel Operator**: Mendeteksi gradien tepi secara horizontal (Sobel X) dan vertikal (Sobel Y).
   - **Prewitt Operator**: Operator berbasis kernel berbobot seragam untuk deteksi kontur tepi.
   - **Roberts Cross Operator**: Operator gradien sederhana berbasis diagonal 2x2.
   - **Canny Edge Detection**: Algoritma multi-tahap (Gaussian filter, gradien, non-maximum suppression, dan hysteresis thresholding).

---

## 🚀 Cara Menjalankan

### 1. Kloning Repositori
```bash
git clone https://github.com/yunikyuroidah/PengolahanCitra.git
cd PengolahanCitra
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
```bash
python -m venv .venv

# Di Windows (Command Prompt / PowerShell):
.venv\Scripts\activate

# Di Linux / macOS:
source .venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Skrip
Pilih salah satu skrip yang ingin dijalankan, misalnya:
```bash
python sobel.py
python canny.py
python ipf8.py
python hpf8.py
```
Hasil pemrosesan akan otomatis tersimpan di folder `outputs/`.

---

## 📦 Pustaka yang Digunakan
- [OpenCV (opencv-python)](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

---

## 👤 Pengembang
- **Yuni Kyuroidah** ([@yunikyuroidah](https://github.com/yunikyuroidah))
