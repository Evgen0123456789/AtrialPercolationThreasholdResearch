import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import stats
from WorkTools import compute_zone_dice
import json
from datetime import datetime

# Пути к данным
ref = "/home/evgeniy/Рабочий стол/Научная работа/Данные/Датасеты/Dataset001_LA_Wall/labelsTs/Ost.AWT.nrrd"
pre = "/home/evgeniy/Рабочий стол/Научная работа/Данные/Датасеты/Dataset001_LA_Wall/window_predictions/Ost.AWT.nrrd"


def Dice(a, b):
    return 2 * np.sum((a == 1) & (b == 1)) / (np.sum(a) + np.sum(b) + 1e-8)


def bootstrap_ci(data1, data2, metric_func, n_boot=1000, alpha=0.05):
    """Bootstrap confidence interval для разницы метрик"""
    diffs = []
    n = len(data1)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        diffs.append(metric_func(data1[idx], data2[idx]))
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    return ci_low, ci_high


if __name__ == "__main__":
    print("=" * 70)
    print("📊 ОТЧЁТ ПО ВАЛИДАЦИИ СЕГМЕНТАЦИИ И РЕГРЕССИИ ТОЛЩИНЫ")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Загрузка данных
    ref_imag = sitk.ReadImage(ref)
    pre_imag = sitk.ReadImage(pre)
    ref_arr = sitk.GetArrayFromImage(ref_imag)
    pre_arr = sitk.GetArrayFromImage(pre_imag)

    # Область анализа (пересечение масок)
    area = (pre_arr != 0) & (ref_arr != 0)
    ref_vals = ref_arr[area].ravel()
    pre_vals = pre_arr[area].ravel()

    # Базовые статистики
    print("\n📈 БАЗОВЫЕ СТАТИСТИКИ (в области перекрытия)")
    print(f"Количество валидных вокселей: {len(ref_vals):,}")
    print(
        f"Референс:  mean={ref_vals.mean():.3f}, std={ref_vals.std():.3f}, min={ref_vals.min():.3f}, max={ref_vals.max():.3f}")
    print(
        f"Предсказание: mean={pre_vals.mean():.3f}, std={pre_vals.std():.3f}, min={pre_vals.min():.3f}, max={pre_vals.max():.3f}")

    # Метрики корреляции
    print("\n🔗 МЕТРИКИ КОРРЕЛЯЦИИ")
    pearson_r, pearson_p = stats.pearsonr(ref_vals, pre_vals)
    spearman_r, spearman_p = stats.spearmanr(ref_vals, pre_vals)

    # Фехнер (вручную)
    P = (ref_vals - ref_vals.mean()) * (pre_vals - pre_vals.mean())
    C, M = np.sum(P > 0), np.sum(P < 0)
    fechner = (C - M) / (C + M) if (C + M) > 0 else 0

    print(f"Пирсон (линейная):     r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    print(f"Спирмен (ранговая):    ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
    print(f"Фехнер (знаковая):     K = {fechner:.4f}")

    # Метрики согласия (регрессия)
    print("\n📏 МЕТРИКИ СОГЛАСИЯ (регрессия толщины)")
    diff = pre_vals - ref_vals
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    mbe = np.mean(diff)  # Mean Bias Error
    std_diff = np.std(diff)

    print(f"MAE (средняя абсолютная ошибка):  {mae:.4f} мм")
    print(f"RMSE (квадратичная ошибка):       {rmse:.4f} мм")
    print(f"MBE (систематическое смещение):   {mbe:+.4f} мм")
    print(f"Std разницы:                      {std_diff:.4f} мм")

    # Доверительные интервалы для MAE и смещения (bootstrap)
    mae_ci = bootstrap_ci(ref_vals, pre_vals, lambda r, p: np.mean(np.abs(p - r)))
    bias_ci = bootstrap_ci(ref_vals, pre_vals, lambda r, p: np.mean(p - r))
    print(f"95% CI для MAE:  [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}] мм")
    print(f"95% CI для MBE:  [{bias_ci[0]:.4f}, {bias_ci[1]:.4f}] мм")

    # Статистические тесты различий
    print("\n🧪 СТАТИСТИЧЕСКИЕ ТЕСТЫ (проверка гипотезы: распределения идентичны)")

    # Проверка нормальности (на подвыборке для скорости)
    sample_size = min(10000, len(ref_vals))
    idx = np.random.choice(len(ref_vals), sample_size, replace=False)
    norm_ref = stats.shapiro(ref_vals[idx])[1]
    norm_pre = stats.shapiro(pre_vals[idx])[1]
    print(f"Нормальность (Shapiro-Wilk, n={sample_size}): референс p={norm_ref:.3e}, предсказание p={norm_pre:.3e}")

    is_normal = norm_ref > 0.05 and norm_pre > 0.05

    if is_normal:
        t_stat, t_p = stats.ttest_rel(ref_vals, pre_vals)
        print(
            f"✅ Парный t-тест: t={t_stat:.4f}, p={t_p:.2e} {'→ РАЗЛИЧИЯ ЗНАЧИМЫ' if t_p < 0.05 else '→ различия не значимы'}")
    else:
        print("⚠️  Распределения не нормальны → используем непараметрические тесты")

    # Непараметрические тесты (всегда)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(ref_vals, pre_vals)
    ks_stat, ks_p = stats.ks_2samp(ref_vals, pre_vals)
    print(
        f"✅ Тест Уилкоксона: W={wilcoxon_stat:.2f}, p={wilcoxon_p:.2e} {'→ РАЗЛИЧИЯ ЗНАЧИМЫ' if wilcoxon_p < 0.05 else '→ различия не значимы'}")
    print(
        f"✅ KS-тест (распределения): D={ks_stat:.4f}, p={ks_p:.2e} {'→ РАСПРЕДЕЛЕНИЯ РАЗЛИЧАЮТСЯ' if ks_p < 0.05 else '→ распределения схожи'}")

    # Размер эффекта (Cohen's d для парных выборок)
    cohens_d = mbe / std_diff if std_diff > 0 else 0
    print(f"\n📐 Размер эффекта (Cohen's d): {cohens_d:+.3f}")
    print("   Интерпретация: |d|<0.2 — малый, 0.2-0.5 — средний, >0.5 — большой")

    # Bland-Altman: пределы согласия
    loa_lower = mbe - 1.96 * std_diff
    loa_upper = mbe + 1.96 * std_diff
    print(f"\n📊 Bland-Altman: пределы согласия 95%: [{loa_lower:.3f}, {loa_upper:.3f}] мм")

    # Zone Dice
    print("\n🎯 ZONE DICE (по диапазонам толщины)")
    zone_dice = compute_zone_dice(ref_arr, pre_arr, units='mm')
    for zone, score in zone_dice.items():
        print(f"{zone:12s} : {score * 100:6.2f} %")

    # ========================================================================
    # 🆕 ДОБАВЛЕНИЕ: Heatmap попарного Dice для зон
    # ========================================================================
    print("\n🔥 Генерация Heatmap по зонам (Pairwise Dice)...")

    # Настройте границы зон под вашу задачу.
    # Можно использовать фиксированные значения или np.linspace/np.percentile
    n_bins = 5
    max_thick = max(ref_vals.max(), pre_vals.max())
    bin_edges = np.linspace(0, max_thick, n_bins + 1)
    zone_labels = [f"({bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f})" for i in range(n_bins)]

    # Формируем бинарные маски для каждой зоны (фон 0 исключается)
    ref_masks, pre_masks = [], []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        l_th = 0.0 if low == 0 else low  # первая зона: >0, остальные: >low
        ref_masks.append((ref_arr > l_th) & (ref_arr <= high))
        pre_masks.append((pre_arr > l_th) & (pre_arr <= high))

    # Считаем попарный Dice для каждой комбинации (Ref_i vs Pre_j)
    zone_heatmap = np.zeros((n_bins, n_bins))
    for i, rm in enumerate(ref_masks):
        for j, pm in enumerate(pre_masks):
            zone_heatmap[i, j] = Dice(rm, pm)

    # Визуализация Heatmap
    fig_heat, ax_heat = plt.subplots(figsize=(7, 6))
    im = ax_heat.imshow(zone_heatmap, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax_heat.set_xticks(range(n_bins))
    ax_heat.set_yticks(range(n_bins))
    ax_heat.set_xticklabels(zone_labels, rotation=45, ha='right', fontsize=9)
    ax_heat.set_yticklabels(zone_labels, fontsize=9)
    ax_heat.set_xlabel("Зоны предсказания", fontsize=10)
    ax_heat.set_ylabel("Зоны референса", fontsize=10)
    ax_heat.set_title("Heatmap: Pairwise Zone Dice", fontsize=12)

    # Аннотации значений в ячейках
    for i in range(n_bins):
        for j in range(n_bins):
            val = zone_heatmap[i, j]
            ax_heat.text(j, i, f"{val:.3f}", ha="center", va="center",
                         color="white" if val > 0.5 else "black", fontsize=9)

    plt.colorbar(im, ax=ax_heat, label="Dice Score")
    plt.tight_layout()
    plt.savefig("zone_pairwise_heatmap.png", dpi=150, bbox_inches='tight')
    print("🖼️  Heatmap сохранён в 'zone_pairwise_heatmap.png'")
    plt.show()
    # ========================================================================

    # Бинарный Dice и IoU для всей маски
    binary_dice_score = Dice(ref_arr > 0, pre_arr > 0)
    intersection = np.sum((ref_arr > 0) & (pre_arr > 0))
    union = np.sum((ref_arr > 0) | (pre_arr > 0))
    iou = intersection / (union + 1e-8)
    print(f"\n🔲 Бинарный Dice: {binary_dice_score * 100:.2f} %, IoU: {iou * 100:.2f} %")

    # Экспорт результатов
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_voxels": int(np.sum(area)),
        "correlation": {
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_r), "spearman_p": float(spearman_p),
            "fechner_k": float(fechner)
        },
        "regression_metrics": {
            "mae_mm": float(mae), "rmse_mm": float(rmse), "mbe_mm": float(mbe),
            "mae_95ci": [float(x) for x in mae_ci], "mbe_95ci": [float(x) for x in bias_ci]
        },
        "statistical_tests": {
            "shapiro_ref_p": float(norm_ref), "shapiro_pre_p": float(norm_pre),
            "wilcoxon_p": float(wilcoxon_p), "ks_2samp_p": float(ks_p)
        },
        "effect_size": {"cohens_d": float(cohens_d)},
        "bland_altman": {"mean_bias": float(mbe), "loa_95": [float(loa_lower), float(loa_upper)]},
        "zone_dice": {k: float(v) for k, v in zone_dice.items()},
        "binary_metrics": {"dice": float(binary_dice_score), "iou": float(iou)}
    }

    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Результаты сохранены в 'validation_report.json'")

    # Визуализация
    print("\n📈 Генерация графиков...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Scatter + регрессия
    ax = axes[0, 0]
    ax.scatter(ref_vals, pre_vals, s=0.2, alpha=0.15, label='Воксели')
    max_val = max(ref_vals.max(), pre_vals.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='y = x (идеал)')
    slope, intercept, r_val, _, _ = stats.linregress(ref_vals, pre_vals)
    x_line = np.linspace(ref_vals.min(), ref_vals.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'g-', linewidth=2, label=f'Регрессия: y={slope:.2f}x+{intercept:.2f}')
    ax.set_xlabel("Толщина: референс (мм)")
    ax.set_ylabel("Толщина: предсказание (мм)")
    ax.set_title(f"Scatter + регрессия (r={pearson_r:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Bland-Altman
    ax = axes[0, 1]
    mean_vals = (ref_vals + pre_vals) / 2
    ax.scatter(mean_vals, diff, s=0.2, alpha=0.15)
    ax.axhline(mbe, color='red', linestyle='--', linewidth=1.5, label=f'Смещение: {mbe:+.3f} мм')
    ax.axhline(loa_upper, color='gray', linestyle=':', label='95% LoA')
    ax.axhline(loa_lower, color='gray', linestyle=':')
    ax.set_xlabel("Средняя толщина (мм)")
    ax.set_ylabel("Разница: предсказание − референс (мм)")
    ax.set_title("Bland-Altman plot")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Гистограммы распределений
    ax = axes[1, 0]
    bins = np.linspace(0, max(ref_vals.max(), pre_vals.max()), 50)
    ax.hist(ref_vals, bins=bins, alpha=0.6, label='Референс', density=True)
    ax.hist(pre_vals, bins=bins, alpha=0.6, label='Предсказание', density=True)
    ax.set_xlabel("Толщина (мм)")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title("Распределение толщины")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Zone Dice bar plot
    ax = axes[1, 1]
    zones = list(zone_dice.keys())
    scores = [v * 100 for v in zone_dice.values()]
    colors = ['#2ecc71' if s > 40 else '#f39c12' if s > 20 else '#e74c3c' for s in scores]
    ax.bar(zones, scores, color=colors, edgecolor='black')
    ax.set_xlabel("Диапазон толщины (мм)")
    ax.set_ylabel("Zone Dice (%)")
    ax.set_title("Качество по зонам толщины")
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.axhline(40, color='green', linestyle='--', alpha=0.5, label='Порог 40%')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("validation_plots.png", dpi=150, bbox_inches='tight')
    print("🖼️  Графики сохранены в 'validation_plots.png'")
    plt.show()

    print("\n✅ Анализ завершён.")