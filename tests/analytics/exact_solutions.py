"""
Аналитические (точные) решения задачи о толщине стенки.

Каждая аналитика — это "эталонная геометрия": для неё толщина в любой внутренней
точке известна замкнуто, вместе с расстояниями до обеих границ.  Такие функции
используются автотестом пайплайна вычисления толщины: по аналитике строится
синтетическая маска стенки (epi/endo), прогоняется через конвейер, и полученное
поле толщины сравнивается с точным ответом.

Конвенции (важно для переносимости тестов между реализациями):
  * Координаты x, y, z — физические, в мм.  Оси массива numpy: [z, y, x]
    (как в SimpleITK-массиве GetArrayFromImage).
  * Для точки p внутри материала:
        thickness(p)      -- полная толщина стенки в этой точке
        dist_epi(p)       -- расстояние от p до эпикарда (внешней границы)
        dist_endo(p)      -- расстояние от p до эндокарда (внутренней границы)
    и по построению выполняется dist_epi + dist_endo == thickness.
  * is_wall(p) возвращает True, если точка лежит в материале стенки.

Список аналитик:
  1. parallel_slab   -- плоская стенка конечной толщины d (точное решение: t = d,
                        расстояния до поверхностей x и d - x).
  2. wedge           -- двугранный угол раствора A: t = R*A, расстояния до граней
                        R*(A-T) и R*T, где R -- расстояние до ребра, T -- угол,
                        под которым видна проекция точки на грань из проекции на
                        ребро (ортогональные проекции).
  3. spherical_shell -- полый шар: t = R_out - R_in (для сферических слоёв
                        толщина не зависит от точки слоя).
  4. cylindrical_shell -- полый цилиндр (бесконечный): t = R_out - R_in.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Общее устройство аналитики
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityZone:
    """Зона интереса для ВЗВЕШЕННОЙ оценки ошибки (см. tests.analytics.quality).

    Смысл: ошибка конвейера неоднородна по объёму, поэтому сравнивать её "по
    всем вокселям" некорректно.  Каждая аналитика сама знает, где метод обязан
    работать лучше всего, и объявляет такие зоны:
      * диск/шар/цилиндр -- центральный цилиндр вокруг оси (там нормали почти
        перпендикулярны решётке, кривизна мала, ступенчатость границы минимальна);
      * двугранный угол -- центральный сектор (подальше от ребра, где t -> 0, и
        подальше от боковых торцов, где граница разрывна).

    Поля:
        name   -- идентификатор зоны для отчёта;
        role   -- "core" (основная, по ней выносится вердикт) или "outer"
                  (дополнительная, только для диагностики);
        weight -- желаемый вес зоны при объединении метрик (обычно доля объёма);
        region -- f(X, Y, Z) -> bool массив той же формы, что grid_fields(bbox).
    """
    name: str
    role: str
    region: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    weight: float = 1.0


@dataclass(frozen=True)
class AnalyticSolution:
    """Точное решение: геометрия + замкнутые формулы толщины и расстояний.

    Поля:
        name        -- имя для отчётов/имён файлов;
        bbox        -- ((x0,x1),(y0,y1),(z0,z1)) физическая область расчёта, мм;
        is_wall     -- f(points Nx3) -> bool N, принадлежность материалу стенки;
        thickness   -- f(points Nx3) -> float N, точная толщина;
        dist_epi    -- f(points Nx3) -> float N, точное расстояние до эпикарда;
        dist_endo   -- f(points Nx3) -> float N, точное расстояние до эндокарда;
        params      -- параметры геометрии (для вывода в отчёт);
        tolerance_rel -- относительный допуск сравнения конвейера с эталоном;
        abs_tol_mm  -- абсолютный допуск, мм (учитывает квантование решётки);
        zones       -- список QualityZone: взвешенные области оценки;
        core_zone   -- имя зоны, по которой принимается главный вердикт.
    """
    name: str
    bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    is_wall: Callable[[np.ndarray], np.ndarray]
    thickness: Callable[[np.ndarray], np.ndarray]
    dist_epi: Callable[[np.ndarray], np.ndarray]
    dist_endo: Callable[[np.ndarray], np.ndarray]
    params: dict
    tolerance_rel: float = 0.15
    abs_tol_mm: float = 0.75
    zones: Tuple[QualityZone, ...] = ()
    core_zone: str = ""

    def summary(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}: {p}"

    def zone_masks(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
        """Маски зон на заданной решётке (X, Y, Z из grid_fields)."""
        return {q.name: np.asarray(q.region(X, Y, Z), dtype=bool) for q in self.zones}

    def tolerances(self, spacing_mm: float) -> Tuple[float, float]:
        """Допуски (относительный, абсолютный в мм) для данного шага решётки.

        Абсолютный допуск заведомо не может быть меньше половины шага: даже
        идеальный метод на дискретной маске не различает границу точнее, чем на
        полвоксела.  Поэтому abs_tol = max(abs_tol_mm, 0.75 * spacing).
        """
        return self.tolerance_rel, max(self.abs_tol_mm, 0.75 * spacing_mm)


def grid_fields(bbox, spacing_xyz):
    """Физические координаты узлов решётки в порядке осей массива [z, y, x].

    Возвращает (X, Y, Z, shape); spacing_xyz = (dx, dy, dz) в мм.

    Узлы располагаются симметрично относительно центра bbox: это важно для
    зон качества -- "центральный" цилиндр/сектор должен проходить ровно через
    геометрический центр фантома независимо от чётности размера.
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox
    dx, dy, dz = spacing_xyz

    def axis(lo, hi, h):
        n = int(np.floor((hi - lo) / h + 1e-9)) + 1     # число узлов
        c = 0.5 * (lo + hi)
        if n % 2 == 1:                                   # нечётное -> узел в центре
            start = c - 0.5 * (n - 1) * h
        else:                                            # чётное -> центр между узлами
            start = c - (0.5 * n - 0.5) * h - 0.5 * h
        return np.arange(n, dtype=np.float64) * h + start

    xs, ys, zs = axis(x0, x1, dx), axis(y0, y1, dy), axis(z0, z1, dz)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return X, Y, Z, X.shape


def grid_center(bbox) -> Tuple[float, float, float]:
    """Геометрический центр области расчёта (x, y, z), мм."""
    (x0, x1), (y0, y1), (z0, z1) = bbox
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)


def _grid_points(bbox, spacing) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Физические координаты узлов решётки (N x 3, порядок колонок x,y,z) и форма."""
    X, Y, Z, shape = grid_fields(bbox, spacing)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts, shape


# ---------------------------------------------------------------------------
# 1. Плоская стенка конечной толщины (идеальная плоскость)
# ---------------------------------------------------------------------------

def parallel_slab(d: float = 4.0,
                  tilt_deg: float = 0.0,
                  center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                  extent_xy: Tuple[float, float] = (60.0, 60.0),
                  margin: float = 8.0) -> AnalyticSolution:
    """Стенка между двумя параллельными плоскостями n·(p-c)=0 и n·(p-c)=d.

    Точное решение: толщина везде равна d; для точки на расстоянии x от одной
    поверхности расстояние до неё равно x, до другой -- d - x.

    tilt_deg -- наклон нормали в плоскости (x, z): при 0 нормаль совпадает с z,
    при ненулевом наклоне плоскости "кажутся" решётке ступенчатыми, и это
    отдельная проверка устойчивости конвейера к дискретизации границы.
    Ограничение по x,y нужно только чтобы маска была конечной; ответ от x,y не
    зависит, поэтому торцы не портят сравнение (внутренние узлы вдали от них).
    """
    half_x, half_y = extent_xy[0] / 2.0, extent_xy[1] / 2.0
    a = np.deg2rad(tilt_deg)
    nx, nz = np.sin(a), np.cos(a)
    cx, cy, cz = center

    def signed(p):
        return (p[:, 0] - cx) * nx + (p[:, 2] - cz) * nz

    def is_wall(p):
        s = signed(p)
        return (s >= 0.0) & (s <= d) & \
               (np.abs(p[:, 0] - cx) <= half_x) & (np.abs(p[:, 1] - cy) <= half_y)

    def dist_epi(p):  # эпикард -- плоскость s = 0
        return signed(p)

    def dist_endo(p):  # эндокард -- плоскость s = d
        return d - signed(p)

    def thickness(p):
        return np.full(p.shape[0], d)

    bbox = ((cx - half_x - margin, cx + half_x + margin),
            (cy - half_y - margin, cy + half_y + margin),
            (cz - half_x * abs(nx) - margin, cz + half_x * abs(nx) + d + margin))

    # Зоны качества для пластины: торцы по x/y -- места разрыва границы, их надо
    # исключать; чем дальше от торцов, тем ближе метод к идеалу.
    def _core(X, Y, Z):
        return ((np.abs(X - cx) <= 0.45 * half_x) & (np.abs(Y - cy) <= 0.45 * half_y))

    def _mid(X, Y, Z):
        return ((np.abs(X - cx) <= 0.75 * half_x) & (np.abs(Y - cy) <= 0.75 * half_y))

    return AnalyticSolution(
        name=f"slab_d{d:g}_t{tilt_deg:g}",
        bbox=bbox,
        is_wall=is_wall,
        thickness=thickness,
        dist_epi=dist_epi,
        dist_endo=dist_endo,
        params={"d": d, "tilt_deg": tilt_deg, "extent_xy": extent_xy},
        tolerance_rel=0.10,
        abs_tol_mm=max(0.6, 0.75 * d / 4.0),
        zones=(QualityZone("central", "core", _core, weight=1.0),
               QualityZone("middle", "outer", _mid, weight=0.5)),
        core_zone="central",
    )


# ---------------------------------------------------------------------------
# 2. Двугранный угол (wedge)
# ---------------------------------------------------------------------------

def wedge(apex_angle_deg: float = 60.0,
          r_min: float = 20.0,
          r_max: float = 45.0,
          length_y: float = 60.0,
          bisector_direction: Tuple[float, float] = (1.0, 0.0),
          margin: float = 8.0) -> AnalyticSolution:
    """Двугранный угол раствора A, ребро -- прямая, параллельная оси y.

    Точное решение: для точки на расстоянии R от ребра толщина t = R * A,
    расстояния до граней: R * (A - T) и R * T, где T -- угол (в той же системе
    отсчёта), под которым из проекции на ребро видна проекция точки на грань
    (ортогональные проекции).

    Материал стенки -- кольцевой сектор r_min <= R <= r_max, |y| <= length_y/2:
    он ограничен (иначе маска бесконечна), а формула t = R*A верна в каждой
    его точке; торцы по y -- плоские границы, которые конвейер тоже обязан
    учитывать, поэтому сравнение ведётся только с внутренними узлами.

    Угловая координата phi измеряется от биссектрисы угла в плоскости (x, z);
    грань phi = -A/2 считается эпикардом, phi = +A/2 -- эндокардом.
    """
    a = np.deg2rad(apex_angle_deg)
    bx, bz = bisector_direction
    bn = np.hypot(bx, bz)
    bx, bz = bx / bn, bz / bn
    # перпендикуляр к биссектрисе в плоскости (x, z)
    tx_, tz_ = -bz, bx

    def polar(p):
        u = p[:, 0] * bx + p[:, 2] * bz
        v = p[:, 0] * tx_ + p[:, 2] * tz_
        return np.hypot(u, v), np.arctan2(v, u)  # R, phi in (-pi, pi]

    half = a / 2.0
    ly = length_y / 2.0

    def is_wall(p):
        r, phi = polar(p)
        return (r >= r_min) & (r <= r_max) & (np.abs(phi) <= half) & \
               (np.abs(p[:, 1]) <= ly)

    def dist_epi(p):  # грань phi = -A/2: расстояние R * T, T = phi + A/2
        r, phi = polar(p)
        return r * (phi + half)

    def dist_endo(p):  # грань phi = +A/2: расстояние R * (A - T)
        r, phi = polar(p)
        return r * (half - phi)

    def thickness(p):
        r, _ = polar(p)
        return r * a

    bbox = ((-r_max - margin, r_max + margin),
            (-ly - margin, ly + margin),
            (-r_max - margin, r_max + margin))

    # Зоны качества для двугранного угла -- ЦЕНТРАЛЬНЫЙ СЕКТОР:
    #  * по углу: держаться подальше от обеих граней (у граней нормаль почти
    #    параллельна поверхности -> ступенчатая дискретизация максимальна);
    #  * по R: не слишком близко к ребру (там t -> 0 и относительная ошибка
    #    взрывается) и не у внешнего торца сектора;
    #  * по y: вдали от боковых торцов, где граница разрывна.
    def _sector(phi_frac, r_lo, r_hi, y_frac):
        def region(X, Y, Z):
            u = X * bx + Z * bz
            v = X * tx_ + Z * tz_
            r = np.hypot(u, v)
            phi = np.arctan2(v, u)
            return ((np.abs(phi) <= phi_frac * half) & (r >= r_lo) & (r <= r_hi)
                    & (np.abs(Y) <= y_frac * ly))
        return region

    r_mid_lo = r_min + 0.25 * (r_max - r_min)
    r_mid_hi = r_min + 0.75 * (r_max - r_min)
    zones = (
        QualityZone("central_sector", "core",
                    _sector(0.45, r_mid_lo, r_mid_hi, 0.5), weight=1.0),
        QualityZone("sector", "outer",
                    _sector(0.65, r_min + 0.1 * (r_max - r_min), r_max, 0.75), weight=0.5),
    )
    return AnalyticSolution(
        name=f"wedge_A{apex_angle_deg:g}_R{r_min:g}-{r_max:g}",
        bbox=bbox,
        is_wall=is_wall,
        thickness=thickness,
        dist_epi=dist_epi,
        dist_endo=dist_endo,
        params={"A_deg": apex_angle_deg, "r_min": r_min, "r_max": r_max,
                "length_y": length_y},
        tolerance_rel=0.15,
        abs_tol_mm=max(0.75, 0.5 * r_min * a * 0.1),  # ~5% толщины на внутр. радиусе
        zones=zones,
        core_zone="central_sector",
    )


# ---------------------------------------------------------------------------
# 3. Полый шар (тонкий слой)
# ---------------------------------------------------------------------------

def spherical_shell(r_in: float = 25.0,
                    thickness_value: float = 4.0,
                    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                    margin: float = 8.0) -> AnalyticSolution:
    """Полый шар: слой r_in <= R <= r_in + t вокруг сферы.

    Точное решение: толщина слоя постоянна и равна t = R_out - R_in;
    расстояния до поверхностей: R - R_in (до внутренней) и R_out - R (до внешней).
    """
    r_out = r_in + thickness_value
    cx, cy, cz = center

    def radius(p):
        return np.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2 + (p[:, 2] - cz) ** 2)

    def is_wall(p):
        r = radius(p)
        return (r >= r_in) & (r <= r_out)

    def dist_epi(p):  # эпикард -- внешняя сфера
        return r_out - radius(p)

    def dist_endo(p):  # эндокард -- внутренняя сфера
        return radius(p) - r_in

    def thickness(p):
        return np.full(p.shape[0], thickness_value)

    # Пояс берётся только в окрестности слоя: лимит по радиусу -- r_out + margin,
    # но не весь шар целиком (иначе узлы внутри полости участвовали бы в зонах).
    lim = r_out + margin
    bbox = ((cx - lim, cx + lim), (cy - lim, cy + lim), (cz - lim, cz + lim))

    # Для шара "центральный цилиндр" вырождается: вырезаем два пояса --
    # околоэкваториальные зоны осей x и y, где нормаль перпендикулярна одной из
    # осей решётки (дискретизация границы минимальна).
    def _belt(coord):
        """Пояс около плоскости, перпендикулярной данной оси: |координата| <= tol*r.

        Внутри такого пояса нормаль к сферам почти параллельна оси решётки,
        поэтому граница слоя дискретизуется ступенькой минимальной высоты.
        """
        def region(X, Y, Z):
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
            return np.abs(coord(X, Y, Z)) <= 0.30 * np.maximum(r, 1e-9)
        return region

    zones = (
        QualityZone("belt_x", "core", _belt(lambda X, Y, Z: X - cx), weight=1.0),
        QualityZone("belt_y", "outer", _belt(lambda X, Y, Z: Y - cy), weight=0.5),
        QualityZone("belt_z", "outer", _belt(lambda X, Y, Z: Z - cz), weight=0.5),
    )
    return AnalyticSolution(
        name=f"ssphere_t{thickness_value:g}_R{r_in:g}",
        bbox=bbox,
        is_wall=is_wall,
        thickness=thickness,
        dist_epi=dist_epi,
        dist_endo=dist_endo,
        params={"r_in": r_in, "t": thickness_value, "center": center},
        tolerance_rel=0.15,
        abs_tol_mm=max(0.75, 0.25 * thickness_value),
        zones=zones,
        core_zone="belt_x",
    )


# ---------------------------------------------------------------------------
# 4. Полый бесконечный цилиндр (ось -- y)
# ---------------------------------------------------------------------------

def cylindrical_shell(r_in: float = 25.0,
                      thickness_value: float = 4.0,
                      axis: int = 1,
                      length_along_axis: float = 60.0,
                      margin: float = 8.0) -> AnalyticSolution:
    """Полый цилиндр (сечение -- кольцо): t = R_out - R_in постоянно."""
    r_out = r_in + thickness_value
    radial_axes = tuple(a for a in range(3) if a != axis)
    la = length_along_axis / 2.0

    def radius(p):
        u, v = p[:, radial_axes[0]], p[:, radial_axes[1]]
        return np.hypot(u, v)

    def is_wall(p):
        r = radius(p)
        return (r >= r_in) & (r <= r_out) & (np.abs(p[:, axis]) <= la)

    def dist_epi(p):
        return r_out - radius(p)

    def dist_endo(p):
        return radius(p) - r_in

    def thickness(p):
        return np.full(p.shape[0], thickness_value)

    bbox_list = []
    for a in range(3):
        if a == axis:
            bbox_list.append((-la - margin, la + margin))
        else:
            bbox_list.append((-r_out - margin, r_out + margin))
    # Для цилиндра естественная зона -- ЦЕНТРАЛЬНЫЙ ЦИЛИНДР вдоль оси: сечение
    # кольцо, но оценка берётся только там, где ось координат проходит через
    # слой (нормаль параллельна оси решётки) и вдали от торцов по оси.
    ra, rb = radial_axes
    la_ = length_along_axis / 2.0

    def _cyl_core(X, Y, Z):
        P = {0: X, 1: Y, 2: Z}
        u, v = P[ra], P[rb]
        w = P[axis]
        r = np.hypot(u, v)
        # узкие "палки" вдоль радиальных осей: |поперёк| <= 0.3 * r
        near_axis = (np.minimum(np.abs(u), np.abs(v)) <= 0.3 * np.maximum(r, 1e-9))
        return near_axis & (np.abs(w) <= 0.5 * la_)

    def _cyl_mid(X, Y, Z):
        P = {0: X, 1: Y, 2: Z}
        w = P[axis]
        return np.abs(w) <= 0.6 * la_

    zones = (
        QualityZone("central_cylinder", "core", _cyl_core, weight=1.0),
        QualityZone("mid_span", "outer", _cyl_mid, weight=0.5),
    )
    return AnalyticSolution(
        name=f"csphere_t{thickness_value:g}_R{r_in:g}",
        bbox=tuple(bbox_list),
        is_wall=is_wall,
        thickness=thickness,
        dist_epi=dist_epi,
        dist_endo=dist_endo,
        params={"r_in": r_in, "t": thickness_value, "axis": axis},
        tolerance_rel=0.15,
        abs_tol_mm=max(0.75, 0.25 * thickness_value),
        zones=zones,
        core_zone="central_cylinder",
    )


# ---------------------------------------------------------------------------
# Реестр аналитик
# ---------------------------------------------------------------------------

def all_analytic_solutions(fast: bool = False) -> list:
    """Все встроенные аналитические решения (расширяйте этот реестр).

    fast=True -- уменьшенный набор для быстрого прогона (одна аналитика на
    семейство геометрий); используется в CI и в smoke-тесте конвейера.
    """
    if fast:
        return [
            parallel_slab(d=4.0),
            wedge(apex_angle_deg=60.0, r_min=20.0, r_max=45.0, length_y=50.0),
            spherical_shell(r_in=25.0, thickness_value=6.0),
        ]
    return [
        parallel_slab(d=4.0),
        parallel_slab(d=2.0),
        parallel_slab(d=4.0, tilt_deg=20.0),
        wedge(apex_angle_deg=60.0, r_min=20.0, r_max=45.0, length_y=50.0),
        wedge(apex_angle_deg=90.0, r_min=20.0, r_max=45.0, length_y=50.0),
        spherical_shell(r_in=25.0, thickness_value=6.0),
        cylindrical_shell(r_in=25.0, thickness_value=6.0, length_along_axis=50.0),
    ]


def core_registry() -> list:
    """Реестр "якорных" аналитик для golden-метрик (стабильный во времени набор).

    Изменять этот список можно только осознанно: сдвиг golden-значений означает
    изменение поведения пайплайна, а не просто добавление новой геометрии.
    """
    return [
        parallel_slab(d=4.0),
        parallel_slab(d=4.0, tilt_deg=20.0),
        wedge(apex_angle_deg=60.0, r_min=20.0, r_max=45.0, length_y=50.0),
        spherical_shell(r_in=25.0, thickness_value=6.0),
        cylindrical_shell(r_in=25.0, thickness_value=6.0, length_along_axis=50.0),
    ]
