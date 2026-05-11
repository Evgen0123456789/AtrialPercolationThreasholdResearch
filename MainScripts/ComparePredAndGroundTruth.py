import os
import SimpleITK as sitk
import pandas as pd

from Tools.WorkTools import get_largest_domain, Dice, IOU

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Программа для сравнения разметок врача и нейросети."
                                                 "Подразумевается, что в разметках есть только маска.")

    parser.add_argument(
        "-gt",
        "--reference",
        type=str,
        help="Путь до папки с врачебной разметкой"
    )
    parser.add_argument(
        "-p",
        "--prediction",
        type=str,
        help="Путь до папки с автоматической разметкой"
    )
    args = parser.parse_args()

    gt_path = args.reference
    p_path = args.prediction

    assert os.path.isdir(gt_path), "GT should be a directory!"
    assert os.path.isdir(p_path), "Prediction path should be a directory!"

    names = set(os.listdir(gt_path)).intersection(set(os.listdir(p_path)))
    # print(names)
    # for name in names:
    #     print(name, os.path.isfile(os.path.join(gt_path, name)), os.path.isfile(os.path.join(p_path, name)))
    names = {name for name in names
             if os.path.isfile(os.path.join(gt_path, name)) and os.path.isfile(os.path.join(p_path, name))}
    assert len(names), "Empty common file set for both dirs!"

    result = {
        "Sample": [],
        "DICE metric": [],
        "IOU metric": []
    }
    for name in names:
        reference = sitk.ReadImage(os.path.join(gt_path, name))
        prediction = sitk.ReadImage(os.path.join(p_path, name))

        gt = sitk.GetArrayFromImage(reference)
        p, _ = get_largest_domain(sitk.GetArrayFromImage(prediction))
        gt, p = gt > 0, p > 0

        result["Sample"].append(name)
        result["DICE metric"].append(Dice(gt, p))
        result["IOU metric"].append(IOU(gt, p))
    df = pd.DataFrame(result)
    print(df)
    df.to_csv('../Results/Comparison_of_manual_and_automatic_segmentation.csv',
              index=False,
              encoding='utf-8-sig')



