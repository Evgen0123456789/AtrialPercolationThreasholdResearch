import os
import numpy as np
import SimpleITK as sitk
from skimage import measure
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support


def load_nrrd(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr, spacing, origin, direction, img


def save_nrrd(img, path):
    sitk.WriteImage(img, path)


def main(dataset_path, name):
    image_path = f"{dataset_path}/imagesTs/{name}_0000.nrrd"
    label_path = f"{dataset_path}/labelsTs/{name}.nrrd"
    pred_path = f"{dataset_path}/predictions/{name}.nrrd"

    print(f"[INFO] Loading: {image_path}")
    vol_arr, spacing, origin, direction, vol_img = load_nrrd(image_path)

    print(f"[INFO] Loading: {label_path}")
    lbl_arr, _, _, _, lbl_img = load_nrrd(label_path)

    print(f"[INFO] Loading: {pred_path}")
    pred_arr, _, _, _, pred_img = load_nrrd(pred_path)

    out_dir = f"{dataset_path}/output/{name}"
    os.makedirs(out_dir, exist_ok=True)

    # Save image
    save_nrrd(vol_img, f"{out_dir}/volume.nrrd")

    # Save 2 labelmaps as-is
    save_nrrd(lbl_img, f"{out_dir}/label_original.seg.nrrd")
    save_nrrd(pred_img, f"{out_dir}/label_pred.seg.nrrd")

    print(f"[OK] Done! Slicer-ready folder: {out_dir}")


if __name__ == "__main__":
    dataset = "Dataset002_LA"
    samples = ["Kar", "S41130", "Sle"]

    dataset_path = "Данные/Датасеты/" + dataset
    for sample in samples:
        main(dataset_path, sample)
