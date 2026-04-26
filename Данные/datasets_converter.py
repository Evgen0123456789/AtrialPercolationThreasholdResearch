import os
import shutil
import numpy as np
import nrrd

source_dir = os.path.join("2018_UTAH_MICCAI", "Testing Set")
images_tr_dir = os.path.join("Датасеты", "Dataset004_LA_MRT_Utah", "imagesTs")
labels_tr_dir = os.path.join("Датасеты", "Dataset004_LA_MRT_Utah", "lablesTs")

os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)

for sample in os.listdir(source_dir):
    sample_dir = os.path.join(source_dir, sample)
    if not os.path.isdir(sample_dir):
        continue

    # ----- Копирование изображения -----
    src_image = os.path.join(sample_dir, "lgemri.nrrd")
    dst_image = os.path.join(images_tr_dir, f"{sample}_0000.nrrd")
    shutil.copy(src_image, dst_image)

    # ----- Загрузка масок -----
    lawall_path = os.path.join(sample_dir, "lawall.nrrd")
    laendo_path = os.path.join(sample_dir, "laendo.nrrd")

    lawall, header = nrrd.read(lawall_path)
    laendo, _ = nrrd.read(laendo_path)

    # ----- Объединение -----
    merged = np.zeros(lawall.shape, dtype=np.uint8)

    merged[lawall == 255] = 1
    merged[laendo == 255] = 2

    # ----- Сохранение -----
    dst_label = os.path.join(labels_tr_dir, f"{sample}.nrrd")
    nrrd.write(dst_label, merged, header)

print("Обработка и копирование завершены.")
