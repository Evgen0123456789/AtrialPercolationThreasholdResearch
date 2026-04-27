#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 03:22:21 2025

@author: evgeniy
"""

import SimpleITK as sitk
import pandas as pd
import numpy as np

def Dice(test: np.ndarray, pred: np.ndarray):
    return 2 * np.sum(test * pred) / (test.sum() + pred.sum())

dataset = "./Данные/Датасеты/Dataset002_LA/"

samples = ["Kar", "S41130", "Sle"]

df = pd.DataFrame(columns=["Sample", "Voluem Dice", "Wall Dice"])

if __name__=="__main__":
    for sample in samples:
        print(sample)
        groundtruth = sitk.ReadImage(dataset + "labelsTs/" + sample + ".nrrd")
        prediction = sitk.ReadImage(dataset + "predictions/" + sample + ".nrrd")
        
        g_arr = sitk.GetArrayFromImage(groundtruth)
        p_arr = sitk.GetArrayFromImage(prediction)
        
        print(np.unique(g_arr), g_arr.shape)
        print(np.unique(p_arr))
        
        if sample == "Kar":
            v_dice = Dice(g_arr[:, :, :, 0], (p_arr == 1).astype(np.uint8))
            w_dice = Dice(g_arr[:, :, :, 1], (p_arr == 2).astype(np.uint8))
        elif sample == "S41130":
            v_dice = Dice((g_arr == 4).astype(np.uint8), (p_arr == 1).astype(np.uint8))
            w_dice = Dice((g_arr == 5).astype(np.uint8), (p_arr == 2).astype(np.uint8))
        elif sample == "Sle":
            v_dice = Dice((g_arr == 1).astype(np.uint8), (p_arr == 1).astype(np.uint8))
            w_dice = Dice((g_arr == 2).astype(np.uint8), (p_arr == 2).astype(np.uint8))
        
        df.loc[len(df)] = {
            "Sample": sample, 
            "Voluem Dice": v_dice,
            "Wall Dice": w_dice
            }
df.set_index("Sample")
df.to_excel('./Отчёты/dice_results_with_voluem.xlsx', index=False)
        