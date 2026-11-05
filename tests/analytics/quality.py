"""
Метрики качества поля толщины и "WARNINGS" для автотеста пайплайна.

Идея: при каждом прогоне конвейера на фантоме считаем не просто ошибку, а
статистику ошибки, из которой потом формируются предупреждения (warnings):

  * STD абсолютной и относительной ошибки -- главные индикаторы стабильности;
    они возвращаются в структуре EvaluationResult и печатаются как WARNINGS,
    если превышают допуск;
  * взвешенные метрики -- усреднение по объёму (voxel volume = произведение
    шага решётки), поэтому доля "тонких" зон не зависит от формы области;
  * weighted-STD по зонам (stratified) -- дисперсия ошибки раскладывается по
    весам зон, что даёт честное объединённое СКО для геометрий, где ошибка
    сильно неоднородна (клиновидный угол, кривые оболочки);
  * зоны интереса задаёт сама аналитика (см. exact_solutions.QualityHint):
    для диска/шара -- центральный цилиндр (осевая зона, где нормаль перпендикулярна
    границам и дискретизация минимальна), для угла -- центральный сектор
    (подальше от ребра и от боковых торцов).

Пороговые значения берутся из AnalyticSolution.tolerance_rel / abs_tol_mm, либо
переопределяются параметром tolerance.  Ничего страшного, если порог превышен:
это WARNING, а не AssertionError -- тест падает только когда warnings содержат
ошибку уровня ERROR (метод вообще не вернул поле).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Уровни предупреждений
# ---------------------------------------------------------------------------

INFO = "INFO"
WARN = "WARN"
ERROR = "ERROR"


@dataclass
class Warning:
    level: str
    code: str
    message: str

    def format(self) -> str:
        return f"[{self.level}] {self.code}: {self.message}"


# ---------------------------------------------------------------------------
# Взвешенная статистика
# ---------------------------------------------------------------------------

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Взвешенное СКО (частотные веса): sqrt(sum w (x-mean)^2 / sum w)."""
    w = weights.sum()
    if w <= 0:
        return float("nan")
    m = np.average(values, weights=weights)
    var = np.average((values - m) ** 2, weights=weights)
    return float(np.sqrt(var))


def stratified_weighted_std(values: np.ndarray,
                            weights: np.ndarray,
                            zones: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, dict]]:
    """Объединённое СКО по зонам + разложение (law of total variance).

    total_var = sum_z p_z * (var_z + (mean_z - grand_mean)^2), p_z = W_z / W.
    Возвращает (объединённое STD, детали по зонам).
    """
    grand_w = weights.sum()
    if grand_w <= 0:
        return float("nan"), {}
    grand_mean = np.average(values, weights=weights)
    total_var = 0.0
    details: Dict[str, dict] = {}
    for name, zmask in zones.items():
        m = zmask & (weights > 0)
        wz = weights[m].sum()
        if wz <= 0:
            details[name] = {"count": int(m.sum()), "weight": 0.0,
                             "mean": float("nan"), "std": float("nan")}
            continue
        vz = values[m]
        wzw = weights[m]
        mean_z = np.average(vz, weights=wzw)
        var_z = np.average((vz - mean_z) ** 2, weights=wzw)
        p_z = wz / grand_w
        total_var += p_z * (var_z + (mean_z - grand_mean) ** 2)
        details[name] = {
            "count": int(m.sum()),
            "weight": float(wz),
            "share": float(p_z),
            "mean": float(mean_z),
            "std": float(np.sqrt(var_z)),
        }
    return float(np.sqrt(max(total_var, 0.0))), details


# ---------------------------------------------------------------------------
# Результат оценки
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Метрики сравнения поля конвейера с эталоном + сформированные warnings."""
    name: str
    n_voxels: int
    voxel_volume_mm3: float
    metrics: Dict[str, float] = field(default_factory=dict)
    zone_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[Warning] = field(default_factory=list)

    # ---- доступ к warnings -------------------------------------------
    def by_level(self, level: str) -> List[Warning]:
        return [w for w in self.warnings if w.level == level]

    @property
    def errors(self) -> List[Warning]:
        return self.by_level(ERROR)

    @property
    def warns(self) -> List[Warning]:
        return self.by_level(WARN)

    def add(self, level: str, code: str, message: str) -> None:
        self.warnings.append(Warning(level, code, message))

    def report(self) -> str:
        lines = [f"=== {self.name} ==="]
        for k, v in self.metrics.items():
            lines.append(f"  {k:>34s}: {v: .4f}")
        for zname, zm in self.zone_metrics.items():
            lines.append(f"  -- zone '{zname}' --")
            for k, v in zm.items():
                lines.append(f"       {k:>28s}: {v: .4f}")
        for w in self.warnings:
            lines.append("  " + w.format())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Основная оценка
# ---------------------------------------------------------------------------

def evaluate_thickness_field(thickness: np.ndarray,
                             reference: np.ndarray,
                             mask: np.ndarray,
                             spacing_xyz: Tuple[float, float, float],
                             zones: Optional[Dict[str, np.ndarray]] = None,
                             *,
                             rel_tol: float = 0.15,
                             abs_tol_mm: float = 0.75,
                             name: str = "",
                             min_valid_voxels: int = 50) -> EvaluationResult:
    """Сравнить поле толщины конвейера с эталонным и собрать warnings.

    thickness/reference/mask -- массивы одинаковой формы, оси [z,y,x];
    mask -- узлы, которые вообще участвуют в сравнении (обычно interior стенки);
    zones -- необязательные именованные подмножества mask (взвешенные метрики
             считаются и по каждой зоне, и объединённое stratified STD);
    spacing_xyz -- (dx, dy, dz), мм; веса = объём вокселя (для равномерного
             сравнения можно передать единичные веса, см. weights ниже).

    Метрики (все в mm, кроме безразмерных):
      count                 число сопоставленных узлов
      ref_mean/ref_std      статистика эталона
      got_mean/got_std      статистика результата конвейера
      bias_mean             среднее (got - ref)  -- систематическое смещение
      std_abs               STD абсолютной ошибки            <-- главный WARN
      std_rel               STD относительной ошибки (доля)  <-- главный WARN
      rmse_abs              sqrt(mean err^2)
      mae_abs               mean |err|
      p95_abs               95-й перцентиль |err|
      frac_out_rel          доля узлов с |err| > rel_tol * ref
      frac_out_abs          доля узлов с |err| > abs_tol_mm
      corr                  корреляция Пирсона got/ref
      coverage              доля mask, где конвейер вернул положительное значение
    """
    res = EvaluationResult(name=name or "thickness", n_voxels=0,
                           voxel_volume_mm3=float(np.prod(spacing_xyz)))

    dx, dy, dz = spacing_xyz
    vox_vol = float(dx * dy * dz)

    mask = mask.astype(bool)
    got = np.asarray(thickness, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    valid = mask & np.isfinite(got) & np.isfinite(ref) & (ref > 0)
    n = int(valid.sum())
    res.n_voxels = n
    if n < min_valid_voxels:
        res.add(ERROR, "NO_DATA",
                f"сопоставлено всего {n} узлов (< {min_valid_voxels}): "
                f"конвейер не вернул поле толщины в этой области")
        res.metrics["count"] = float(n)
        return res

    g, r = got[valid], ref[valid]
    err = g - r
    rel_err = err / r
    weights = np.full(n, vox_vol)  # равные объёмы -> веса эквивалентны числу вокселей

    m = res.metrics
    m["count"] = float(n)
    m["ref_mean"] = weighted_mean(r, weights)
    m["ref_std"] = weighted_std(r, weights)
    m["got_mean"] = weighted_mean(g, weights)
    m["got_std"] = weighted_std(g, weights)
    m["bias_mean"] = weighted_mean(err, weights)
    m["std_abs"] = weighted_std(err, weights)
    m["std_rel"] = weighted_std(rel_err, weights)
    m["rmse_abs"] = float(np.sqrt(np.mean(err ** 2)))
    m["mae_abs"] = float(np.mean(np.abs(err)))
    m["p95_abs"] = float(np.percentile(np.abs(err), 95))
    m["frac_out_rel"] = float(np.mean(np.abs(err) > rel_tol * r))
    m["frac_out_abs"] = float(np.mean(np.abs(err) > abs_tol_mm))
    m["corr"] = float(np.corrcoef(g, r)[0, 1]) if np.std(g) > 0 and np.std(r) > 0 else float("nan")
    m["coverage"] = float(((got > 0) & mask).sum() / max(mask.sum(), 1))

    # ----Warnings---------------------------------------------------------
    _threshold_warnings(res, m, rel_tol=rel_tol, abs_tol_mm=abs_tol_mm)

    # ----Зоны (взвешенные метрики + stratified STD)----------------------
    if zones:
        zone_masks = {}
        for zname, zmask in zones.items():
            zm = np.asarray(zmask, dtype=bool) & valid
            if zm.sum() >= 10:
                zone_masks[zname] = zm
        if zone_masks:
            zg = np.zeros_like(got)
            zr = np.zeros_like(ref)
            ze = np.zeros_like(got)
            zg[valid] = g
            zr[valid] = r
            ze[valid] = err
            zw = np.zeros_like(got)
            zw[valid] = weights

            per_zone_vals = {k: ze[v] for k, v in zone_masks.items()}
            per_zone_w = {k: zw[v] for k, v in zone_masks.items()}
            combined, details = _combined_std(per_zone_vals, per_zone_w)
            m["std_abs_stratified"] = combined

            for zname, zm in zone_masks.items():
                # zv/zr_/zw_ уже плоские (proba по valid): zm -- маска над ними,
                # повторная 3D-индексация невозможна и давала IndexError
                zv = ze[zm]
                zr_ = zr[zm]
                zw_ = zw[zm]
                zm_rel = zv / np.where(zr_ > 0, zr_, np.nan)
                d = details.get(zname, {})
                res.zone_metrics[zname] = {
                    "count": float(d.get("count", zm.sum())),
                    "share": float(d.get("share", np.nan)),
                    "ref_mean": weighted_mean(zr_, zw_),
                    "bias_mean": weighted_mean(zv, zw_),
                    "std_abs": weighted_std(zv, zw_),
                    "std_rel": weighted_std(zm_rel, zw_),
                    "mae_abs": float(np.mean(np.abs(zv))),
                    "p95_abs": float(np.percentile(np.abs(zv), 95)),
                }
                _threshold_warnings(res, res.zone_metrics[zname],
                                    rel_tol=rel_tol, abs_tol_mm=abs_tol_mm,
                                    prefix=f"zone:{zname}:")

    return res


def _combined_std(zone_values: Dict[str, np.ndarray],
                  zone_weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, dict]]:
    """Объединённое (по зонам) взвешенное СКО ошибки."""
    all_v = np.concatenate([zone_values[k] for k in zone_values])
    all_w = np.concatenate([zone_weights[k] for k in zone_values])
    grand = weighted_mean(all_v, all_w)
    total_w = all_w.sum()
    total_var = 0.0
    details = {}
    for k, v in zone_values.items():
        w = zone_weights[k]
        wz = w.sum()
        if wz <= 0:
            continue
        mean_z = weighted_mean(v, w)
        var_z = weighted_std(v, w) ** 2
        p_z = wz / total_w
        total_var += p_z * (var_z + (mean_z - grand) ** 2)
        details[k] = {"count": int(len(v)), "weight": float(wz), "share": float(p_z),
                      "mean": float(mean_z), "std": float(np.sqrt(var_z))}
    return float(np.sqrt(max(total_var, 0.0))), details


def _threshold_warnings(res: EvaluationResult, mm: Dict[str, float], *,
                        rel_tol: float, abs_tol_mm: float,
                        prefix: str = "") -> None:
    """Превышение допусков -> WARNINGS (STD ошибки -- центральная метрика)."""
    tag = prefix

    def val(key):
        return mm.get(key, float("nan"))

    std_abs = val("std_abs")
    std_rel = val("std_rel")
    bias = val("bias_mean")
    frac_out = val("frac_out_rel")
    cov = val("coverage")
    got_mean = val("got_mean")
    ref_mean = val("ref_mean")

    if not np.isfinite(std_abs):
        return

    # 1. Раздутое СКО абсолютной ошибки
    if std_abs > 2.0 * abs_tol_mm:
        res.add(ERROR, f"{tag}STD_ABS_CRITICAL",
                f"STD(abs error) = {std_abs:.3f} mm критически велико "
                f"(порог {2.0 * abs_tol_mm:.3f} mm)")
    elif std_abs > abs_tol_mm:
        res.add(WARN, f"{tag}STD_ABS_HIGH",
                f"STD(abs error) = {std_abs:.3f} mm превышает допуск "
                f"{abs_tol_mm:.3f} mm")

    # 2. Раздутое СКО относительной ошибки
    if std_rel > 4.0 * rel_tol:
        res.add(ERROR, f"{tag}STD_REL_CRITICAL",
                f"STD(rel error) = {std_rel:.3f} критически велико "
                f"(порог {4.0 * rel_tol:.3f})")
    elif std_rel > 2.0 * rel_tol:
        res.add(WARN, f"{tag}STD_REL_HIGH",
                f"STD(rel error) = {std_rel:.3f} выше 2x допуска ({2.0 * rel_tol:.3f})")
    elif std_rel > rel_tol:
        res.add(WARN, f"{tag}STD_REL_ABOVE_TOL",
                f"STD(rel error) = {std_rel:.3f} выше допуска {rel_tol:.3f}")

    # 3. Систематическое смещение
    if abs(bias) > 2.0 * abs_tol_mm:
        res.add(ERROR, f"{tag}BIAS_CRITICAL",
                f"смещение = {bias:+.3f} mm критично (порог {2.0 * abs_tol_mm:.3f} mm)")
    elif abs(bias) > abs_tol_mm:
        res.add(WARN, f"{tag}BIAS_HIGH",
                f"смещение = {bias:+.3f} mm превышает допуск {abs_tol_mm:.3f} mm")

    # 4. Выбросы
    if frac_out > 0.30:
        res.add(WARN, f"{tag}OUTLIERS",
                f"{100 * frac_out:.1f}% узлов вне допуска "
                f"+-{100 * rel_tol:.0f}%")

    # 5. Пропущенные значения / потеря масштаба
    if cov < 0.85:
        res.add(WARN, f"{tag}LOW_COVERAGE",
                f"конвейер вернул толщину лишь для {100 * cov:.1f}% узлов области")
    if np.isfinite(got_mean) and np.isfinite(ref_mean) and ref_mean > 0:
        scale = got_mean / ref_mean
        if scale < 0.5 or scale > 2.0:
            res.add(ERROR, f"{tag}SCALE_WRONG",
                    f"средняя толщина относится к эталону как {scale:.3f} "
                    f"-- похоже на сбой метода/единиц измерения")
        elif abs(scale - 1.0) > rel_tol:
            res.add(WARN, f"{tag}SCALE_OFF",
                    f"средняя толщина относится к эталону как {scale:.3f}")


# ---------------------------------------------------------------------------
# Проброс warnings в python-warnings (опционально: pytest -W error::UserWarning)
# ---------------------------------------------------------------------------

def emit_python_warnings(res: EvaluationResult, *, as_error: bool = False) -> None:
    """Продублировать WARNINGS через модуль warnings (для логов pytest)."""
    for w in res.warnings:
        if w.level == ERROR or (w.level == WARN and as_error):
            warnings.warn(w.format(), RuntimeWarning, stacklevel=2)
        elif w.level == WARN:
            warnings.warn(w.format(), UserWarning, stacklevel=2)
