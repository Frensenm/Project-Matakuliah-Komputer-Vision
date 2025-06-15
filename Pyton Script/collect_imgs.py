import os
import cv2
import numpy as np

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)
start_class = 0  # Ganti dengan angka awal yang diinginkan
for j in range(start_class, start_class + number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        # === Preprocessing ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Ubah ke grayscale
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Mengurangi noise
        edges = cv2.Canny(blurred, 50, 150)  # Deteksi tepi dengan Canny

        # === Overlay Edges di Gambar Asli ===
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Konversi ke BGR agar bisa digabungkan
        overlay = cv2.addWeighted(frame, 0.8, edges_colored, 0.5, 0)  # Gabungkan gambar asli dengan tepi

        # Show frame dengan garis tepi
        cv2.imshow('frame', overlay)
        cv2.waitKey(25)

        # Save the image
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), overlay)

        counter += 1

cap.release()
cv2.destroyAllWindows()
