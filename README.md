# Implementasi Algoritma Greedy dan Template Matching untuk Otomatisasi Pencapaian Skor Maksimal pada Permainan Candy Crush

**Mata Kuliah:** STIMA — Strategi Algoritma  
**Nama:** Dika Pramudya Nugraha  
**NIM:** 13524132

Bot analisis permainan Match-3 yang membaca screenshot papan permainan, mengonversinya menjadi matriks, lalu menentukan langkah swap terbaik menggunakan **algoritma Greedy** dengan simulasi cascade lengkap.

---

## Deskripsi Proyek

Program ini dirancang sebagai implementasi gabungan **Computer Vision** dan **Algoritma Greedy** untuk permainan puzzle Match-3 (seperti Candy Crush). Bot tidak memainkan game secara otomatis di layar, melainkan:

1. Membaca **screenshot** papan permainan
2. Mendeteksi **ukuran grid** dan **jenis permen** secara otomatis
3. Mensimulasikan semua kemungkinan **swap** antar sel bersebelahan
4. Memilih langkah dengan **skor prediksi tertinggi** (termasuk reaksi berantai / cascade)
5. Menampilkan hasil di **GUI**: matriks terdeteksi, panah arah, dan skor

---

## Fitur Utama

- Deteksi otomatis **ROI** (area papan) dan ukuran grid (5×5 s/d 10×10)
- Klasifikasi permen berdasarkan warna **HSV** (OpenCV)
- Deteksi sel kosong / obstacle (`X`) — area air, swirl, frosting
- Simulasi aturan Match-3: match, gravitasi, dan cascade
- Algoritma Greedy untuk pemilihan langkah optimal satu langkah
- GUI interaktif dengan overlay grid, panah arah, dan matriks visual
- Ekspor hasil analisis ke folder `hasil_uji/`
- Unit test untuk logic game dan modul vision

---

## Teknologi

| Komponen | Library / Tool |
|----------|----------------|
| Computer Vision | OpenCV (`opencv-python`) |
| Operasi matriks | NumPy |
| GUI | Tkinter + Pillow |
| Bahasa | Python 3.10+ |

---

## Prasyarat

- Python 3.10 atau lebih baru
- pip (package manager Python)
- Tkinter (biasanya sudah terinstal bersama Python di Windows)

---

## Instalasi

```powershell
cd "d:\Semester 4\STIMA\Makalah\match3_bot"
pip install -r requirements.txt
```

Dependensi utama:

```
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## Cara Menjalankan

### GUI Analisis Screenshot

Simpan screenshot uji di folder `screenshot_uji/` (contoh: `uji_1.png`, `uji_4.png`).

```powershell
python gui_app.py
```

**Langkah penggunaan:**

1. Klik **Pilih Screenshot** — dialog terbuka di folder `screenshot_uji/`
2. Pilih gambar (mis. `uji_1.png` untuk grid 8×8, `uji_4.png` untuk grid 6×7)
3. Klik **Analisis & Cari Langkah** — ukuran grid dideteksi otomatis
4. Lihat hasil: panah arah swap, skor prediksi, koordinat sel, dan matriks permen
5. (Opsional) Klik **Unduh Hasil** untuk menyimpan gambar overlay ke folder `hasil_uji/`
6. Klik **Reset** untuk memulai ulang dari awal

### Unit Test

```powershell
python -m unittest discover tests -v
```

Test mencakup:

- Deteksi match horizontal/vertikal
- Gravitasi dengan obstacle `X`
- Simulasi swap dan cascade
- Algoritma Greedy
- Deteksi otomatis ukuran grid dari screenshot
- Klasifikasi sel permen dan sel kosong

---

## Alur Kerja Program

```
Screenshot (PNG/JPG)
       │
       ▼
  vision.py ──► Deteksi ROI + ukuran grid + klasifikasi sel
       │
       ▼
  Matriks papan (B/H/O/U/P/X)
       │
       ▼
  greedy.py ──► Coba semua swap → pilih skor tertinggi
       │
       │ (memanggil)
       ▼
  game_logic.py ──► find_matches → apply_gravity → cascade
       │
       ▼
  gui_app.py ──► Tampilkan panah arah, skor, matriks visual
```

---

## Representasi Papan

Papan direpresentasikan sebagai matriks 2D berisi karakter:

| Karakter | Arti |
|----------|------|
| `B` | Permen Biru |
| `H` | Permen Hijau |
| `O` | Permen Oranye |
| `U` | Permen Ungu |
| `P` | Permen Pink |
| `X` | Sel kosong / obstacle (tidak bisa di-swap) |
| `?` | Placeholder permen baru setelah gravitasi (simulasi) |

---

## Sistem Skor

| Panjang Match | Skor |
|---------------|------|
| 3 permen | 30 poin |
| 4 permen | 60 poin |
| 5 permen | 100 poin |
| > 5 permen | 100 + (lebihnya × 20) poin |

Skor cascade (match → gravitasi → match lagi) dijumlahkan untuk setiap simulasi swap.

---

## Struktur Proyek

```
match3_bot/
├── game_logic.py          # Engine simulasi Match-3 (match, gravitasi, cascade)
├── greedy.py              # Algoritma Greedy pemilihan langkah terbaik
├── vision.py              # Computer Vision: screenshot → matriks papan
├── gui_app.py             # Antarmuka grafis (Tkinter)
├── requirements.txt       # Dependensi Python
├── README.md              # Dokumentasi proyek
├── screenshot_uji/        # Folder input screenshot uji
│   ├── uji_1.png          # Contoh papan 8×8
│   └── uji_4.png          # Contoh papan 6×7 dengan area kosong
├── hasil_uji/             # Folder output hasil analisis (dibuat otomatis)
└── tests/
    ├── test_logic.py      # Unit test game_logic & greedy
    └── test_vision.py     # Unit test deteksi grid & klasifikasi sel
```

---

## Penjelasan Modul

### `vision.py` — Computer Vision

- `detect_board()` — deteksi otomatis ROI papan dan ukuran grid
- `image_to_board()` — konversi gambar menjadi matriks karakter
- `draw_grid_overlay()` — gambar garis grid dan label di screenshot
- `draw_move_arrow()` — gambar panah arah swap yang disarankan

Klasifikasi sel menggunakan analisis **HSV** (Hue, Saturation, Value) untuk membedakan jenis permen, latar air toska, dan swirl/frosting.

### `game_logic.py` — Engine Simulasi

- `find_matches()` — deteksi match horizontal & vertikal ≥ 3
- `apply_gravity()` — gravitasi permen setelah match (menghormati obstacle `X`)
- `simulate_swap()` — simulasi swap + cascade lengkap, mengembalikan total skor

### `greedy.py` — Algoritma Greedy

- `find_best_move()` — iterasi semua swap ke kanan/bawah, pilih skor tertinggi
- `analyze_board()` — wrapper yang mengembalikan langkah, skor, arah, dan koordinat

### `gui_app.py` — Antarmuka Grafis

Mengintegrasikan semua modul ke dalam GUI Tkinter dengan panel kontrol, preview screenshot, matriks visual berwarna, dan tombol ekspor hasil.

---

## Algoritma Greedy

Greedy memilih langkah terbaik **saat ini** tanpa mempertimbangkan langkah jauh ke depan:

1. Untuk setiap sel yang bukan `X`, coba swap ke **kanan** dan **bawah**
2. Simulasikan hasil swap termasuk **cascade**
3. Hitung **total skor** dari seluruh reaksi berantai
4. Pilih swap dengan skor **tertinggi**

> **Catatan:** Greedy optimal untuk satu langkah, bukan strategi multi-langkah jangka panjang.

---

## Contoh Output

Setelah analisis, GUI menampilkan:

- **Ukuran grid** terdeteksi (mis. `8 baris × 8 kolom`)
- **Arah geser** (↑ ATAS / ↓ BAWAH / ← KIRI / → KANAN)
- **Skor prediksi** dari simulasi cascade
- **Koordinat** sel sumber dan tujuan
- **Matriks visual** berwarna sesuai jenis permen
- **Panah kuning** pada screenshot menunjukkan langkah yang disarankan

---

## Penulis

**Dika Pramudya Nugraha**  
NIM: **13524132**

Proyek makalah mata kuliah STIMA — Strategi Algoritma.
