"""
Собственная (независимая от конвейера) оценка "идеального" метода на фантоме.

Зачем это нужно: прежде чем гонять тяжёлый пайплайн, надо знать, какая часть
расхождения с аналитикой обусловлена ДИСКРЕТИЗАЦИЕЙ (ступенчатая граница маски),
а какая -- самим методом.  Здесь считается то же самое, что делает конвейер, но
по идеальной формуле и на той же решётке:

    t_ideal(node) = EDT(node -> дискретная поверхность эндокарда)
                  + EDT(node -> дискретная поверхность эпикарда)

где EDT -- евклидово distance transform по маске вокселей.  Это и есть
"ответ идеального метода на данном фантоме": расхождение t_ideal с непрерывной
аналитикой и есть неустранимая погрешность дискретизации, которую нельзя ставить
в вину пайплайну.

Дополнительно считается DISCRETE-ЭТАЛОН -- поле t_ideal, которое тест передаёт в
quality.evaluate_thickness_field как reference.  Именно сравнение с ним даёт
честную ошибку метода (bias/STD), потому что обе стороны живут на одной сетке.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

_STRUCT = ndimage.generate_binary_structure(3, 1)


def _edt(set_a: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> np.ndarray:
    """Евклидово расстояние от каждого узла до множества set_a (>0), мм.

    Оси массива -- [z,y,x], поэтому sampling формируется в порядке (dz, dy, dx)
    из spacing_xyz = (dx, dy, dz).
    """
    dx, dy, dz = spacing_xyz
    return ndimage.distance_transform_edt(set_a == 0, sampling=(dz, dy, dx))


@dataclass
class DiscreteReference:
    """Поля эталона, согласованные с ФАКТИЧЕСКОЙ дискретной геометрией фантома."""
    dist_epi: np.ndarray     # расстояние до дискретной поверхности эпикарда
    dist_endo: np.ndarray    # расстояние до дискретной поверхности эндокарда
    thickness: np.ndarray    # их сумма = толщина идеального метода на сетке
    analytic_gap: Dict[str, float]   # расхождение с непрерывной аналитикой


def discrete_reference(wall: np.ndarray,
                       epi: np.ndarray,
                       endo: np.ndarray,
                       exact_thickness: np.ndarray,
                       spacing_xyz: Tuple[float, float, float],
                       interior: np.ndarray | None = None) -> DiscreteReference:
    """Построить дискретный эталон толщины по граничным слоям маски.

    wall/epi/endo -- uint8 маски [z,y,x]; exact_thickness -- непрерывная
    аналитика на той же решётке (для оценки погрешности дискретизации);
    interior -- узлы, где сравниваем (по умолчанию wall без граничных слоёв).
    """
    if interior is None:
        interior = (wall > 0) & (epi == 0) & (endo == 0)

    d_epi = _edt(epi.astype(bool), spacing_xyz)
    d_endo = _edt(endo.astype(bool), spacing_xyz)
    t_disc = d_epi + d_endo

    m = interior.astype(bool)
    ref_t = np.where(m, t_disc, 0.0)

    gap = _gap_stats(ref_t[m], np.asarray(exact_thickness)[m])
    return DiscreteReference(dist_epi=np.where(m, d_epi, 0.0),
                             dist_endo=np.where(m, d_endo, 0.0),
                             thickness=ref_t,
                             analytic_gap=gap)


def _gap_stats(disc: np.ndarray, cont: np.ndarray) -> Dict[str, float]:
    """Насколько дискретный эталон расходится с непрерывной аналитикой."""
    if disc.size == 0:
        return {"count": 0.0}
    err = disc - cont
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = err / np.where(cont > 0, cont, np.nan)
    return {
        "count": float(disc.size),
        "mean_mm": float(np.mean(err)),
        "std_mm": float(np.std(err)),
        "max_abs_mm": float(np.max(np.abs(err))),
        "rel_std": float(np.nanstd(rel)),
    }


def dilate_boundary_layers(mask: np.ndarray, layers: int = 1) -> np.ndarray:
    """Граничные слои маски (для проверки устойчивости к расширению границы)."""
    out = mask.astype(bool).copy()
    struct = _STRUCT
    grown = mask.astype(bool)
    for _ in range(layers):
        grown = ndimage.binary_dilation(grown, structure=struct) & (mask > 0)
        out |= grown
    return out.astype(np.uint8)
