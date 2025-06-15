# Project-Matakuliah-Komputer-Vision

## Deteksi Bahasa Isyarat Indonesia (SIBI) secara Real-Time

Proyek ini merupakan tugas akhir untuk mata kuliah Komputer Visi saya dan 3 orang teman kelompok saya . Tujuan dari proyek ini adalah untuk membangun sebuah sistem yang mampu mendeteksi dan menerjemahkan gestur abjad dari Bahasa Isyarat Indonesia (SIBI) menjadi teks secara real-time menggunakan webcam.

## Fitur Utama & Pengembangan

Proyek ini dibangun berdasarkan fondasi dan referensi dari proyek *Sign language detection with Python and Scikit Learn* oleh Computer vision engineer. Kami mengambil alur kerja dasar tersebut kemudian melakukan **modifikasi dan pengembangan signifikan** pada beberapa tahap kunci:

#### 1\. Preprocessing Data Tingkat Lanjut (Tahap Pengumpulan Data)

Tidak seperti referensi yang menyimpan gambar asli, kami mengimplementasikan beberapa langkah *preprocessing* secara real-time saat data dikumpulkan untuk membantu model lebih fokus pada bentuk dan kontur tangan:

  * **Grayscaling:** Mengubah gambar menjadi skala abu-abu.
  * **Gaussian Blur:** Menghaluskan gambar dan mengurangi *noise*.
  * **Canny Edge Detection:** Mendeteksi tepi atau kontur dari tangan.
  * **Overlay:** Hasil deteksi tepi kemudian digabungkan (di-overlay) kembali ke gambar asli, sehingga menciptakan dataset yang kaya akan informasi bentuk.

#### 2\. Peningkatan Algoritma Klasifikasi (Tahap Pelatihan)

Referensi asli menggunakan algoritma *Random Forest*. Untuk meningkatkan akurasi dan performa, kami menggantinya dengan **XGBoost (Extreme Gradient Boosting)**, sebuah algoritma yang terkenal dengan kecepatan dan kemampuannya dalam menghasilkan model berakurasi tinggi.

#### 3\. Fitur Inferensi yang Interaktif (Tahap Deteksi Real-Time)

Kami menambahkan beberapa fungsionalitas baru pada skrip deteksi untuk menjadikannya lebih dari sekadar pendeteksi huruf:

  * **Penyusun Kalimat:** Aplikasi dapat **menyimpan setiap huruf yang terdeteksi menjadi sebuah kata atau kalimat**, yang ditampilkan di layar.
  * **Fungsi Reset:** Pengguna dapat dengan mudah **menghapus kalimat yang sudah terbentuk** untuk memulai dari awal.
  * **Kamus SIBI Terintegrasi:** Kami menyertakan gambar-gambar dari kamus SIBI sebagai referensi, menjadikannya alat bantu belajar yang praktis bagi mereka yang baru mengenal bahasa isyarat.

## Teknologi yang Digunakan

  * **Python 3.9+**
  * **OpenCV:** Untuk pemrosesan gambar dan video serta interaksi dengan webcam.
  * **MediaPipe:** Untuk deteksi landmark tangan secara real-time.
  * **XGBoost:** Sebagai algoritma utama untuk klasifikasi gestur.
  * **Scikit-learn:** Untuk pembagian dataset dan evaluasi model.
  * **NumPy:** Untuk operasi numerik dan manipulasi data.

## Struktur Proyek

Proyek ini dibagi menjadi 4 skrip utama:

1.  `1_collect_data.py`: Menjalankan webcam untuk mengumpulkan data gambar dengan *preprocessing*.
2.  `2_extract_features.py`: Mengekstrak data landmark dari setiap gambar menggunakan MediaPipe.
3.  `3_train_model.py`: Melatih model klasifikasi XGBoost menggunakan data fitur.
4.  `4_inference.py`: Menjalankan aplikasi deteksi real-time dengan semua fitur interaktif.

## Referensi

Proyek ini tidak akan terwujud tanpa fondasi dan inspirasi dari proyek luar biasa yang dibuat oleh **Computer vision engineer**. Kami mempelajari alur kerjanya dan kemudian mengembangkannya dengan modifikasi dan fitur tambahan dari tim kami.

  * **Video YouTube Referensi:** [Sign language detection with Python and Scikit Learn | Landmark detection | Computer vision tutorial](https://www.youtube.com/watch?v=MJCSjXepaAM)
  * **Repositori GitHub Referensi:** [computervisioneng/sign-language-detector-python](https://github.com/computervisioneng/sign-language-detector-python/)

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detailnya.
