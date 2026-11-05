"""
Генерация синтетических фантомов по аналитическим решениям.

Фантом = маска стенки (wall), две границы (epi -- внешняя, endo -- внутренняя) и
точные поля толщины/расстояний на той же решётке, что использует AWT-конвейер:

  * оси массива numpy: [z, y, x];
  * spacing хранится в порядке sitk (dx, dy, dz);
  * узлы поля = центры вокселей (как после upscale+downscale в GridTools).

Маска строится семплированием функции is_wall(p) в центрах вокселей, поэтому её
граница может "дрожать" на полвоксела относительно идеальной поверхности.  Из-за
этого непрерывная аналитика НЕ является корректным эталоном для поэлементного
сравнения с выходом конвейера: она отличается от расстояния до ФАКТИЧЕСКОЙ
дискретной границы на величину порядка шага решётки.

Поэтому у фантома два эталона:
  * exact_*  -- непрерывная аналитика; служит для контроля самого фантома
                (тождество dist_epi + dist_endo == thickness) и для оценки
                погрешности дискретизации;
  * ref      -- дискретный эталон (EDT до реальных граничных слоёв маски), см.
                tests.analytics.discrete_reference.  Именно с ним сравнивается
                поле конвейера: обе стороны живут на одной сетке, и остаточная
                ошибка -- это ошибка МЕТОДА, а не дискретизации.

Дополнительно фантом несёт zone_masks -- взвешенные зоны качества, объявленные
самой аналитикой (центральный цилиндр для диска/шара, центральный сектор для
угла); метод zones_for_field приводит их к решётке поля конвейера.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from tests.analytics.discrete_reference import DiscreteReference, discrete_reference
from tests.analytics.exact_solutions import AnalyticSolution, grid_fields

_STRUCT = ndimage.generate_binary_structure(3, 1)


# ---------------------------------------------------------------------------
# Фантом
# ---------------------------------------------------------------------------

@dataclass
class Phantom:
    name: str
    wall: np.ndarray      # uint8 маска стенки, оси [z,y,x]
    epi: np.ndarray       # uint8, воксели стенки у внешней границы
    endo: np.ndarray      # uint8, воксели стенки у внутренней границы
    cavity: np.ndarray    # uint8, полость (самая большая компонента дополнения)
    spacing_xyz: Tuple[float, float, float]   # (dx, dy, dz), мм
    origin_xyz: Tuple[float, float, float]    # физкая координата центра вокселя [0,0,0]
    exact_thickness: np.ndarray   # float64 [z,y,x], непрерывная аналитика (0 вне стенки)
    exact_dist_epi: np.ndarray    # float64 [z,y,x]
    exact_dist_endo: np.ndarray   # float64 [z,y,x]
    analytic: AnalyticSolution
    ref: DiscreteReference = None                     # дискретный эталон (EDT)
    zone_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def interior(self) -> np.ndarray:
        """Воксели стенки, не лежащие на границах (их должен считать ПДЭ)."""
        return (self.wall > 0) & (self.epi == 0) & (self.endo == 0)

    @property
    def reference_thickness(self) -> np.ndarray:
        """Эталон для сравнения с конвейером (дискретный, согласован с маской)."""
        if self.ref is None:
            return self.exact_thickness
        return self.ref.thickness

    def core_zone_mask(self) -> np.ndarray:
        """Маска основной (core) зоны качества; если зоны нет -- весь interior."""
        z = self.zone_masks.get(self.analytic.core_zone)
        if z is None:
            return self.interior
        return self.interior & z

    def zones_for_field(self,
                        shape: Tuple[int, int, int],
                        scale: int = 1,
                        shift: Tuple[int, int, int] = (0, 0, 0)) -> Dict[str, np.ndarray]:
        """Привести зоны качества к решётке поля конвейера.

        scale  -- во сколько раз шаг поля конвейера мельче шага фантома
                  (по умолчанию фантом генерируется с тем же шагом, что поле,
                  тогда scale=1);
        shift  -- срез окна: индексы поля конвейера смещены относительно
                  индексов фантома (bounding box + strip + downscale).
        """
        out = {}
        for k, m in self.zone_masks.items():
            zm = np.asarray(m, dtype=np.uint8)
            if scale != 1:
                zm = np.repeat(np.repeat(np.repeat(zm, scale, 0), scale, 1), scale, 2)
            out[k] = _crop_or_pad(zm, shape, shift).astype(bool)
        return out


def _crop_or_pad(arr: np.ndarray, shape, shift) -> np.ndarray:
    """Обрезать/дополнить нулями массив arr до shape со сдвигом shift (z,y,x)."""
    res = np.zeros(shape, dtype=arr.dtype)
    src_off = [max(0, -s) for s in shift]
    dst_off = [max(0, s) for s in shift]
    extent = [min(arr.shape[a] - src_off[a], shape[a] - dst_off[a]) for a in range(3)]
    if any(e <= 0 for e in extent):
        return res
    sl_src = tuple(slice(src_off[a], src_off[a] + extent[a]) for a in range(3))
    sl_dst = tuple(slice(dst_off[a], dst_off[a] + extent[a]) for a in range(3))
    res[sl_dst] = arr[sl_src]
    return res


# ---------------------------------------------------------------------------
# Построение
# ---------------------------------------------------------------------------

def _field(fn, pts: np.ndarray, shape) -> np.ndarray:
    """Скалярное поле аналитики на решётке."""
    return fn(pts).reshape(shape).astype(np.float64)


def make_phantom(analytic: AnalyticSolution,
                 spacing_mm: float = 1.0,
                 anisotropy: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Phantom:
    """Дискретизировать аналитическое решение на регулярной решётке."""
    dx, dy, dz = tuple(s * a for s, a in zip((spacing_mm,) * 3, anisotropy))
    X, Y, Z, shape = grid_fields(analytic.bbox, (dx, dy, dz))
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    wall = analytic.is_wall(pts).reshape(shape).astype(np.uint8)

    t_exact = np.where(wall > 0, _field(analytic.thickness, pts, shape), 0.0)
    d_epi = np.where(wall > 0, _field(analytic.dist_epi, pts, shape), 0.0)
    d_endo = np.where(wall > 0, _field(analytic.dist_endo, pts, shape), 0.0)

    # ---- Полость и границы --------------------------------------------
    outside_all = wall == 0
    lab, n_comp = ndimage.label(outside_all, structure=_STRUCT)
    if n_comp > 0:
        sizes = ndimage.sum(np.ones_like(lab, dtype=np.int64), lab, index=range(1, n_comp + 1))
        cavity_mask = lab == (int(np.argmax(sizes)) + 1)
    else:
        cavity_mask = np.zeros(shape, dtype=bool)

    near_cavity = ndimage.binary_dilation(cavity_mask, structure=_STRUCT) & (wall > 0)
    outer_region = outside_all & ~cavity_mask
    near_outer = ndimage.binary_dilation(outer_region, structure=_STRUCT) & (wall > 0)

    if near_cavity.any() and near_outer.any():
        endo = near_cavity.astype(np.uint8)
        epi = near_outer.astype(np.uint8)
    else:
        # Нет замкнутой полости (пластина): границу делим пополам вдоль оси z.
        both = near_cavity | near_outer
        zs = np.arange(shape[0])[:, None, None]
        mid = shape[0] // 2
        endo = (both & (zs >= mid)).astype(np.uint8)
        epi = (both & (zs < mid)).astype(np.uint8)

    origin_xyz = (float(X[0, 0, 0]), float(Y[0, 0, 0]), float(Z[0, 0, 0]))

    ph = Phantom(
        name=analytic.name,
        wall=wall, epi=epi, endo=endo, cavity=cavity_mask.astype(np.uint8),
        spacing_xyz=(dx, dy, dz),
        origin_xyz=origin_xyz,
        exact_thickness=t_exact,
        exact_dist_epi=d_epi,
        exact_dist_endo=d_endo,
        analytic=analytic,
    )

    # ---- Дискретный эталон и зоны качества ---------------------------
    ph.ref = discrete_reference(wall, epi, endo, t_exact, (dx, dy, dz), ph.interior)
    ph.zone_masks = analytic.zone_masks(X, Y, Z)
    return ph


# ---------------------------------------------------------------------------
# Проверки самих аналитик (без конвейера)
# ---------------------------------------------------------------------------

def self_check(analytic: AnalyticSolution, n_samples: int = 200_000, seed: int = 0) -> float:
    """Сэмплировать точки внутри материала и проверить тождества решения.

    Возвращает максимальную невязку |dist_epi + dist_endo - thickness|.
    """
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1), (z0, z1) = analytic.bbox
    pts = np.column_stack([
        rng.uniform(x0, x1, n_samples),
        rng.uniform(y0, y1, n_samples),
        rng.uniform(z0, z1, n_samples),
    ])
    inside = analytic.is_wall(pts)
    pts = pts[inside]
    assert len(pts) > 100, f"слишком мало точек внутри {analytic.name}"
    t = analytic.thickness(pts)
    de = analytic.dist_epi(pts)
    dn = analytic.dist_endo(pts)
    assert np.all(t > 0) and np.all(de >= 0) and np.all(dn >= 0), \
        f"отрицательные значения в {analytic.name}"
    return float(np.abs(de + dn - t).max())


def phantom_consistency(ph: Phantom) -> dict:
    """Диагностика фантома: сколько узлов, насколько тождество выполнено на сетке."""
    interior = ph.interior
    resid = np.abs(
        ph.exact_dist_epi[interior] + ph.exact_dist_endo[interior] - ph.exact_thickness[interior]
    )
    gap = ph.ref.analytic_gap if ph.ref is not None else {}
    core = ph.core_zone_mask()
    return {
        "wall_voxels": int(ph.wall.sum()),
        "interior_voxels": int(interior.sum()),
        "core_zone_voxels": int(core.sum()),
        "max_identity_residual": float(resid.max()) if interior.sum() else 0.0,
        "mean_identity_residual": float(resid.mean()) if interior.sum() else 0.0,
        # расхождение дискретного эталона с непрерывной аналитикой:
        # это погрешность ДИСКРЕТИЗАЦИИ, её нельзя ставить в вину методу
        "discretization_mean_mm": gap.get("mean_mm", float("nan")),
        "discretization_std_mm": gap.get("std_mm", float("nan")),
        "discretization_max_mm": gap.get("max_abs_mm", float("nan")),
    }
