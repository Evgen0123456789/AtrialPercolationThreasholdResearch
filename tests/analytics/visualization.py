"""
Визуализация автотеста толщины: "чтобы можно было глазами посмотреть".

Инструменты -- те, что уже есть в проекте (matplotlib из Tools/PlotTools.py и
SimpleITK из Tools/SaveTools.py).  VTK здесь НЕ используется (vtkmodules в CI
отсутствует); вместо него поля пишутся в .nrrd, которые удобно смотреть в
MITK / 3D Slicer.

Что рисуется (для каждого фантомного кейса):

  1. Фантом сам по себе (без конвейера) -- `render_phantom`:
     мозаика срезов wall/epi/endo/cavity + тепловые карты exact_thickness,
     dist_epi, dist_endo с контурами зон качества (core-зона жирным).
     Это контроль ДО того, как мы вообще что-то считаем.

  2. Результат прогона конвейера -- `render_pipeline_result`:
     для выбранных срезов в один ряд: маска стенки -> эталон t -> поле
     конвейера -> карта ошибки (расходящиеся "биполярные" цвета) -> гистограмма
     ошибок; плюс отдельная панель cross-section профилей толщины вдоль линии,
     пересекающей стенку (там видно, где метод проваливается: у торцов, у ребра,
     на кривизне).

  3. Сводка всех кейсов -- `render_summary_dashboard`:
     таблица метрик + по одному профилю/карте ошибки на кейс; warnings
     (WARN/ERROR из quality.evaluate_thickness_field) подписаны красным прямо
     под своим кейсом, чтобы глаз сразу цеплял проблемные геометрии.

  4. HTML-галерея -- `write_html_gallery` / `render_case_report`:
     одна страница со всеми PNG и списком WARNINGS; открывается одним кликом.

Все функции чистые (ничего не пишут вне output-директории) и не требуют
дисплея (backend Agg), поэтому работают в CI.  Ключевые кадры дополнительно
сохраняются в `tests/reports/golden/` -- это визуальные golden-артефакты: при
изменении пайплайна картинка меняется, и `compare_images` даёт числовую
оценку отличия (см. tests/test_visualization.py).
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe: вызывать ДО импорта pyplot
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from tests.analytics.phantom_generator import Phantom  # noqa: E402
from tests.analytics.quality import EvaluationResult, ERROR, WARN  # noqa: E402

# ---------------------------------------------------------------------------
# Конвенции
# ---------------------------------------------------------------------------
# Оси numpy-массивов во всём тесте: [z, y, x].  Срез по индексу a фиксирует
# эту ось; оставшиеся две рисуются как (горизонталь, вертикаль).  Для a=0 (z)
# это (y, x) -- аксиальная плоскость; для a=1 (y) -- (z, x) сагиттальная;
# для a=2 (x) -- (z, y) корональная.
PLANE_NAMES = {0: "axial (const z)", 1: "sagittal (const y)", 2: "coronal (const x)"}


def _plane_axes(slice_axis: int) -> Tuple[int, int]:
    """(horiz, vert) -- оси данных, отложенные по X и Y图 для данного среза."""
    others = [a for a in range(3) if a != slice_axis]
    return others[0], others[1]


def physical_extent(arr_shape, spacing_xyz, slice_axis, origin_xyz=None):
    """Параметры plt.imshow для среза: пиксели в МИЛЛИМЕТРАХ, а не в номерах вокселей.

    Возвращает dict с полями:
      * imshow_kw -- extent/origin/aspect для plt.imshow(**imshow_kw);
      * contour_kw -- transform для ax.contour(..., **contour_kw). extent задаёт
        только imshow; contour принимает координаты данных, поэтому ему нужен
        AffineTransform (иначе контуры зон ложатся на пиксельную сетку и
        "плывут" относительно поля).
    Без этого картинки выглядят правильно, но подписи осей были бы бесполезны:
    на глаз важно видеть, где именно в мм случилась ошибка.
    """
    from matplotlib.transforms import Affine2D

    horiz, vert = _plane_axes(slice_axis)
    # Массивы лежат в порядке осей [z, y, x], а spacing_xyz/origin_xyz заданы в
    # физическом порядке (x, y, z): numpy-ось a <-> spacing_xyz[2 - a].  Без
    # этого пересчёта на анизотропной решётке масштаб по осям был бы перепутан.
    sp = tuple(spacing_xyz[2 - a] for a in range(3))
    o = ((0.0, 0.0, 0.0) if origin_xyz is None
         else tuple(float(origin_xyz[2 - a]) for a in range(3)))
    n_h = arr_shape[horiz]
    n_v = arr_shape[vert]
    left = o[horiz] - 0.5 * sp[horiz]
    right = o[horiz] + (n_h - 0.5) * sp[horiz]
    # индекс vert растёт вниз на изображении -> физически меняем местами low/high
    bottom = o[vert] + (n_v - 0.5) * sp[vert]
    top = o[vert] - 0.5 * sp[vert]
    # пиксельная сетка (col,row) -> физическая плоскость (h-ось, v-ось)
    tr = (Affine2D()
          .scale(sp[horiz], -sp[vert])
          .translate(o[horiz] - 0.5 * sp[horiz], o[vert] + 0.5 * sp[vert]))
    return dict(imshow_kw=dict(extent=(left, right, bottom, top), origin="upper", aspect="equal"),
                contour_kw=dict(transform=tr))


def pick_representative_slices(phantom: Phantom,
                               n_per_axis: int = 2,
                               prefer_core: bool = True) -> List[Tuple[int, int]]:
    """Выбрать "говорящие" срезы: те, где стенки больше всего и где есть core-зона.

    Возвращает список (slice_axis, index).  Простой перебор всех трёх осей с
    ранжированием по числу interior-вокселей; при prefer_core добав бонус за
    наличие основной зоны качества (именно там вердикт теста).
    """
    interior = phantom.interior
    core = phantom.core_zone_mask()
    picks: List[Tuple[int, int]] = []
    for axis in range(3):
        counts = interior.sum(axis=tuple(a for a in range(3) if a != axis))
        if counts.max() <= 0:
            continue
        if prefer_core and core.any():
            ccounts = core.sum(axis=tuple(a for a in range(3) if a != axis))
            score = counts + 0.5 * ccounts  # бонус за зону вердикта
        else:
            score = counts.astype(float)
        order = np.argsort(score)[::-1]
        taken: List[int] = []
        for idx in order:
            if score[idx] <= 0:
                break
            # не брать соседние срезы одного пика -- иначе дубли картинок
            if all(abs(int(idx) - t) >= max(1, interior.shape[axis] // 8) for t in taken):
                taken.append(int(idx))
            if len(taken) >= n_per_axis:
                break
        picks.extend((axis, i) for i in taken)
    return picks


def _sl(axis: int, idx: int):
    return (slice(None),) * axis + (idx,) + (slice(None),) * (2 - axis)


def _contour_zones(ax, zones: Dict[str, np.ndarray], slice_axis: int, idx: int,
                   ext: dict, colors: Optional[Dict[str, str]] = None):
    """Контуры зон качества поверх тепловой карты; core-зона -- жирным.

    ext -- dict из physical_extent(): contour рисуется в пиксельных координатах
    среза через AffineTransform (contour_kw), поэтому он совпадает с imshow,
    у которого физическая сетка задана extent'ом.
    """
    colors = colors or {}
    ck = ext["contour_kw"] if "contour_kw" in ext else {}
    for name, mask in zones.items():
        m = np.asarray(mask, dtype=bool)[_sl(slice_axis, idx)]
        if m.sum() < 3:
            continue
        is_core = "core" in name.lower() or "central" in name.lower()
        ax.contour(m.astype(float), levels=[0.5],
                   colors=[colors.get(name, "k" if not is_core else "m")],
                   linewidths=(2.2 if is_core else 0.9),
                   linestyles=("-" if is_core else "--"),
                   **ck)
        # подпись зоны -- около её центра масс (в тех же пиксельных координатах)
        ys, xs = np.nonzero(m)
        if len(ys):
            px, py = ck["transform"].transform((xs.mean(), ys.mean())) if ck else (xs.mean(), ys.mean())
            ax.text(px, py, name, fontsize=6.5, alpha=0.85,
                    color=colors.get(name, "k" if not is_core else "m"),
                    ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.55, pad=0.6))


def _extent_tuple(ext: dict) -> Tuple[float, float, float, float]:
    if "extent" in ext:                      # старый формат: плоский dict imshow-параметров
        l, r, b, t = ext["extent"]
        return (l, r, b, t)
    e = ext["imshow_kw"]["extent"]
    return (e[0], e[1], e[2], e[3])


# ---------------------------------------------------------------------------
# Палитры
# ---------------------------------------------------------------------------
CMAP_THICKNESS = "viridis"      # толщина/расстояния: последовательная
CMAP_MASK = "Greys"             # бинарные маски
CMAP_ERROR = "coolwarm"         # ошибка: расходящаяся, 0 = белый


def _norm_for(values: np.ndarray, symmetric: bool = False):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    if symmetric:
        m = max(abs(lo), abs(hi), 1e-9)
        return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    if hi - lo < 1e-12:
        return None
    return None  # пусть matplotlib сам масштабирует


# ---------------------------------------------------------------------------
# 1. Визуализация фантома (до прогона конвейера)
# ---------------------------------------------------------------------------
def render_phantom(phantom: Phantom,
                   out_dir: str,
                   slices: Optional[Sequence[Tuple[int, int]]] = None,
                   dpi: int = 130) -> Dict[str, str]:
    """Контроль самого фантома: маска, границы, точные поля, зоны качества.

    Возвращает {"png": путь, "nrrd": путь} (nrrd может отсутствовать, если
    SimpleITK недоступен).
    """
    os.makedirs(out_dir, exist_ok=True)
    slices = list(slices) if slices else pick_representative_slices(phantom, n_per_axis=1)
    zones = phantom.zone_masks

    fields = [
        ("wall", phantom.wall, CMAP_MASK, False),
        ("epi", phantom.epi, CMAP_MASK, False),
        ("endo", phantom.endo, CMAP_MASK, False),
        ("cavity", phantom.cavity, CMAP_MASK, False),
        ("exact thickness", phantom.exact_thickness, CMAP_THICKNESS, False),
        ("dist_epi", phantom.exact_dist_epi, CMAP_THICKNESS, False),
        ("dist_endo", phantom.exact_dist_endo, CMAP_THICKNESS, False),
    ]

    ncol = len(fields)
    nrow = max(1, len(slices))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow), squeeze=False)

    for row, (axis, idx) in enumerate(slices):
        ext = physical_extent(phantom.wall.shape, phantom.spacing_xyz, axis,
                              phantom.origin_xyz)
        for col, (title, arr, cmap, sym) in enumerate(fields):
            ax = axes[row][col]
            slc = np.asarray(arr, dtype=float)[_sl(axis, idx)]
            im = ax.imshow(slc, cmap=cmap, **ext["imshow_kw"])
            if title.endswith("thickness") or title.startswith("dist"):
                _contour_zones(ax, zones, axis, idx, ext)
                # граница стенки -- тонкий белый контур, чтобы видеть связь поля и формы
                wall_slc = (phantom.wall > 0)[_sl(axis, idx)]
                ax.contour(wall_slc.astype(float), levels=[0.5], colors=["w"],
                           linewidths=0.4, **ext["contour_kw"])
            ax.set_title(f"{title}\n{PLANE_NAMES[axis]} @ {idx}", fontsize=7.5)
            ax.set_xticks([]); ax.set_yticks([])
            if title != "wall" and cmap == CMAP_THICKNESS:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Phantom self-check: {phantom.name}   "
                 f"spacing={phantom.spacing_xyz[0]:g}mm   "
                 f"voxels(z,y,x)={phantom.wall.shape}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(out_dir, f"phantom_{_safe(phantom.name)}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    out = {"png": png}
    nrrd = write_fields_nrrd(phantom, os.path.join(out_dir, f"phantom_{_safe(phantom.name)}.nrrd"))
    if nrrd:
        out["nrrd"] = nrrd
    _write_sidecar(png, phantom=phantom.name, slices=slices,
                   note="phantom self-check: exact fields + quality zones")
    return out


# ---------------------------------------------------------------------------
# 2. Визуализация результата конвейера
# ---------------------------------------------------------------------------
@dataclass
class CaseRender:
    """Результат отрисовки одного кейса (для golden-сравнения и галереи)."""
    name: str
    png: str
    profile_png: str = ""
    metrics_png: str = ""
    nrrd: str = ""
    slices: List[Tuple[int, int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def render_pipeline_result(phantom: Phantom,
                           pipeline_thickness: np.ndarray,
                           evaluation: Optional[EvaluationResult] = None,
                           out_dir: str = "tests/reports/figs",
                           slices: Optional[Sequence[Tuple[int, int]]] = None,
                           reference: Optional[np.ndarray] = None,
                           compare_exact: bool = False,
                           dpi: int = 130) -> CaseRender:
    """Главная картинка: стенка | эталон | конвейер | ошибка (+профили).

    pipeline_thickness -- поле той же формы, что phantom.wall (оси [z,y,x]);
      если форма отличается, ожидается, что вызывающий привёл её (см.
      Phantom.zones_for_field / align_to_phantom).
    reference -- чем сравнивать; по умолчанию phantom.reference_thickness
      (дискретный EDT-эталон).  compare_exact=True заставляет взять непрерывную
      аналитику phantom.exact_thickness -- полезно, чтобы глазами оценить
      погрешность дискретизации (она видна как "рамка" у границы).
    evaluation -- если передан, его warnings печатаются под картинкой и
      попадают в CaseRender.warnings (для галереи/золота).
    """
    os.makedirs(out_dir, exist_ok=True)
    got = np.asarray(pipeline_thickness, dtype=np.float64)
    if got.shape != phantom.wall.shape:
        raise ValueError(f"shape mismatch: field {got.shape} vs phantom {phantom.wall.shape}")

    ref = np.asarray(reference if reference is not None else
                     (phantom.exact_thickness if compare_exact
                      else phantom.reference_thickness), dtype=np.float64)
    valid = phantom.interior & (ref > 0)
    err = got - ref

    slices = list(slices) if slices else pick_representative_slices(phantom, n_per_axis=1)
    if not slices:
        slices = [(0, phantom.wall.shape[0] // 2)]

    panels = [
        ("wall + zones", lambda a, i: (phantom.wall > 0)[_sl(a, i)].astype(float), CMAP_MASK, False),
        ("reference t [mm]", lambda a, i: _masked(ref, valid, a, i), CMAP_THICKNESS, False),
        ("pipeline t [mm]", lambda a, i: _masked(got, valid, a, i), CMAP_THICKNESS, False),
        ("error t_pipe - t_ref [mm]", lambda a, i: _masked(err, valid, a, i), CMAP_ERROR, True),
    ]

    nrow = len(slices)
    fig, axes = plt.subplots(nrow, len(panels) + 1, figsize=(2.6 * (len(panels) + 1), 2.7 * nrow),
                             squeeze=False)
    for row, (axis, idx) in enumerate(slices):
        ext = physical_extent(phantom.wall.shape, phantom.spacing_xyz, axis, phantom.origin_xyz)
        for col, (title, fn, cmap, sym) in enumerate(panels):
            ax = axes[row][col]
            data = fn(axis, idx)
            norm = _norm_for(data, symmetric=sym) if sym else None
            vmin = vmax = None
            if sym and norm is None:
                m = np.nanmax(np.abs(data[np.isfinite(data)])) if np.isfinite(data).any() else 1.0
                vmin, vmax = -max(m, 1e-9), max(m, 1e-9)
            im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                          interpolation="nearest", **ext["imshow_kw"])
            if col == 0:
                _contour_zones(axes[row][col], phantom.zone_masks, axis, idx, ext)
            if col == 3:  # ошибка: поверх контур core-зоны -- вердикт по ней
                _contour_zones(ax, phantom.zone_masks, axis, idx, ext)
            ax.set_title(f"{title}\n{PLANE_NAMES[axis]} @ {idx}", fontsize=7.5)
            ax.set_xticks([]); ax.set_yticks([])
            if cmap != CMAP_MASK:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # последняя колонка -- гистограмма ошибок на этом срезе
        ax = axes[row][len(panels)]
        e = err[_sl(axis, idx)][valid[_sl(axis, idx)]]
        if e.size:
            lim = max(np.percentile(np.abs(e), 99), 1e-6)
            ax.hist(e, bins=41, range=(-lim, lim), color="steelblue", edgecolor="k", lw=0.3)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"error hist (slice)\nmean={np.mean(e):+.3f} std={np.std(e):.3f} mm",
                         fontsize=7.5)
        else:
            ax.set_title("error hist: no data", fontsize=7.5)
        ax.set_xlabel("mm", fontsize=7)

    tag = "exact" if compare_exact else "discrete-ref"
    fig.suptitle(f"{phantom.name}  [{tag}]  "
                 f"interior voxels={int(valid.sum())}, core={int(phantom.core_zone_mask().sum())}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    png = os.path.join(out_dir, f"case_{_safe(phantom.name)}_{tag}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    prof = render_profiles(phantom, got, ref, valid, os.path.join(out_dir, png), dpi=dpi)
    metrics_png = ""
    if evaluation is not None:
        metrics_png = render_metrics_panel(evaluation,
                                           os.path.join(out_dir, f"metrics_{_safe(phantom.name)}.png"),
                                           dpi=dpi)
    nrrd = write_field_nrrd(got, phantom, os.path.join(out_dir, f"pipeline_{_safe(phantom.name)}.nrrd"))

    warns = [w.format() for w in (evaluation.warnings if evaluation else [])]
    _write_sidecar(png, phantom=phantom.name, slices=slices, tag=tag, warnings=warns)
    return CaseRender(name=phantom.name, png=png, profile_png=prof, metrics_png=metrics_png,
                      nrrd=nrrd or "", slices=slices, warnings=warns)


def _masked(arr: np.ndarray, valid: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Срез с NaN вне valid -- тогда фон панели прозрачный, ошибка не 'размазывается'."""
    d = np.asarray(arr, dtype=float)[_sl(axis, idx)].copy()
    v = valid[_sl(axis, idx)]
    d[~v] = np.nan
    return d


def render_profiles(phantom: Phantom,
                    got: np.ndarray,
                    ref: np.ndarray,
                    valid: np.ndarray,
                    base_png: str,
                    dpi: int = 130) -> str:
    """Профили толщины "насквозь": как выглядит стенка в разрезе глазами.

    Для каждой из трёх осей берётся центральная плоскость (median-полоса в 3
    среза, чтобы подавить шум ступенчатой дискретизации) и строится профиль
    вдоль оставшихся двух координат.  На одном графике -- reference (чёрный) и
    pipeline (красный): плато = d у пластины, линейный рост R*A у клина, провалы
    видны прямо у торцов/ребра.  Это самый быстрый способ увидеть, ГДЕ именно
    метод врёт, а не только НАСКОЛЬКО.
    """
    shape = phantom.wall.shape
    # numpy-ось a <-> spacing_xyz[2 - a] (порядок массива [z, y, x])
    spacing_by_axis = [phantom.spacing_xyz[2 - a] for a in range(3)]
    origin_by_axis = [phantom.origin_xyz[2 - a] for a in range(3)]
    axis_names = {0: "z", 1: "y", 2: "x"}  # имя по НUMPY-оси массива
    fig, axes = plt.subplots(1, 3, figsize=(3.6 * 3, 2.9), squeeze=False)
    for k, axis in enumerate(range(3)):
        ax = axes[0][k]
        horiz, vert = _plane_axes(axis)     # данные среза: [vert, horiz]
        c = shape[axis] // 2
        band = range(max(0, c - 1), min(shape[axis], c + 2))

        def prof(arr, msk, keep_axis):
            """Профиль толщины вдоль одной из осей плоскости среза.

            Срез фиксированной оси имеет форму [shape[vert], shape[horiz]] (т.е.
            ось 0 внутри среза -- это vert, ось 1 -- horiz).  Усредняем по
            ПРОТИВОПОЛОЖНОЙ оси и возвращаем профиль длиной shape[data_axis], где
            data_axis = (vert, horiz)[keep_axis].  Толщина от оси усреднения у
            всех аналитик не зависит, значит среднее корректно; NaN -- где нет
            валидных вокселей.
            """
            avg_axis = 1 - keep_axis
            n_prof = arr[_sl(axis, band[0])].shape[keep_axis]
            vals = np.full(n_prof, np.nan)
            for i in band:
                a = arr[_sl(axis, i)]
                m = msk[_sl(axis, i)] & (a > 0)
                num = np.where(m, a, 0.0).sum(axis=avg_axis)
                den = m.sum(axis=avg_axis)
                cur = np.full(num.shape, np.nan)
                ok = den > 0
                cur[ok] = num[ok] / den[ok]
                vals = np.fmin(vals, cur)   # худшее из полосы: видно локальные провалы
            return vals

        # два независимых разреза на каждую фиксированную ось: профиль вдоль
        # horiz и профиль вдоль vert (каждый со своим цветом/подписью)
        styles = [("tab:blue", "tab:cyan"), ("tab:red", "orange")]
        plotted = False
        for j, (keep_axis, data_axis) in enumerate(((1, horiz), (0, vert))):
            pr = prof(ref, valid, keep_axis)
            pg = prof(got, valid, keep_axis)
            coord = (np.arange(pr.size) * spacing_by_axis[data_axis]
                     + origin_by_axis[data_axis])       # длина из формы самого среза
            ok = np.isfinite(pr)
            if not ok.any():
                continue
            plotted = True
            c_ref, c_got = styles[j]
            ax.plot(coord[ok], pr[ok], color=c_ref, lw=1.6,
                    label=f"reference ~ {axis_names[data_axis]}")
            ax.plot(coord[ok], pg[ok], color=c_got, lw=1.1, alpha=0.9, ls="--",
                    label=f"pipeline ~ {axis_names[data_axis]}")
        ax.set_title(f"profiles in plane {PLANE_NAMES[axis]}\n(band of 3 slices @ idx {c})",
                     fontsize=8)
        ax.set_xlabel("mm", fontsize=7); ax.set_ylabel("t [mm]", fontsize=7)
        ax.grid(alpha=0.3)
        if plotted:                       # пустой legend() печатает UserWarning
            ax.legend(fontsize=5.8, loc="best")
    fig.suptitle(f"cross-section profiles: {phantom.name}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(os.path.dirname(base_png), "profile_" + os.path.basename(base_png))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def render_metrics_panel(evaluation: EvaluationResult, out_png: str, dpi: int = 130) -> str:
    """Одна панель: метрики + WARNING-статистика (STD -- крупно)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axis("off")
    lines = [f"{k}: {v:.4f}" for k, v in evaluation.metrics.items()]
    txt = "\n".join(lines) or "(no metrics)"
    ax.text(0.0, 0.98, txt, fontsize=7.0, va="top", family="monospace")
    warn_txt = "\n".join(w.format() for w in evaluation.warnings) or "no warnings"
    color = "crimson" if evaluation.errors else ("darkorange" if evaluation.warns else "green")
    ax.text(0.0, 0.30, warn_txt, fontsize=6.6, va="top", color=color, family="monospace")
    ax.set_title(f"WARNINGS: {evaluation.name}", fontsize=9, color=color)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 3. Сводный дашборд по всем кейсам
# ---------------------------------------------------------------------------
def render_summary_dashboard(cases: Iterable[CaseRender],
                             out_png: str,
                             evaluations: Optional[Dict[str, EvaluationResult]] = None,
                             dpi: int = 120) -> str:
    """Сетка: по строке на кейс -- error-map | профиль | таблица метрик/warnings."""
    cases = list(cases)
    evaluations = evaluations or {}
    ncol = 3
    fig, axes = plt.subplots(len(cases), ncol, figsize=(3.3 * ncol, 2.9 * max(1, len(cases))),
                             squeeze=False)
    for r, case in enumerate(cases):
        # (a) error map первого доступного среза
        ax = axes[r][0]
        _show_png_thumb(ax, case.png, f"{case.name}: result")
        # (b) профили
        ax = axes[r][1]
        _show_png_thumb(ax, case.profile_png, "profiles ref/pipeline")
        # (c) текстовая сводка
        ax = axes[r][2]
        ax.axis("off")
        ev = evaluations.get(case.name)
        body = []
        if ev is not None:
            keys = ["count", "bias_mean", "std_abs", "std_rel", "mae_abs", "p95_abs",
                    "frac_out_rel", "corr", "coverage"]
            body += [f"{k}={ev.metrics.get(k, float('nan')):.4f}" for k in keys if k in ev.metrics]
            for zn, zm in list(ev.zone_metrics.items())[:3]:
                body.append(f"[{zn}] bias={zm.get('bias_mean', float('nan')):+.3f} "
                            f"std={zm.get('std_abs', float('nan')):.3f}")
        body += case.warnings[:6]
        color = "crimson" if (ev and ev.errors) else ("darkorange" if (ev and ev.warns or case.warnings) else "black")
        ax.text(0.0, 0.98, "\n".join(body) or "(no data)", fontsize=6.2, va="top",
                family="monospace", color=color)
        ax.set_title(f"summary: {case.name}", fontsize=8, color=color)
    fig.suptitle("Thickness autotest dashboard (visual review)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _show_png_thumb(ax, path: str, title: str):
    if path and os.path.exists(path):
        from matplotlib.image import imread
        ax.imshow(imread(path))
    else:
        ax.text(0.5, 0.5, "missing image", ha="center", va="center", color="gray")
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# 4. Экспорт в NRRD (для MITK/Slicer) -- тем же инструментом, что SaveTools
# ---------------------------------------------------------------------------
def write_field_nrrd(field_arr: np.ndarray, phantom: Phantom, path: str) -> Optional[str]:
    try:
        import SimpleITK as sitk
    except Exception:
        return None
    img = sitk.GetImageFromArray(np.asarray(field_arr, dtype=np.float32))
    img.SetSpacing(tuple(float(v) for v in phantom.spacing_xyz))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    ox, oy, oz = phantom.origin_xyz
    img.SetOrigin((ox, oy, oz))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sitk.WriteImage(img, path)
    return path


def write_fields_nrrd(phantom: Phantom, path: str) -> Optional[str]:
    """Мульти-канал: wall|epi|endo|cavity|exact_t|ref_t (как в SaveTools)."""
    try:
        import SimpleITK as sitk
    except Exception:
        return None
    stack = np.stack([
        phantom.wall.astype(np.float32),
        phantom.epi.astype(np.float32),
        phantom.endo.astype(np.float32),
        phantom.cavity.astype(np.float32),
        phantom.exact_thickness.astype(np.float32),
        np.asarray(phantom.reference_thickness, dtype=np.float32),
    ], axis=0)
    img = sitk.GetImageFromArray(stack)
    img.SetSpacing(tuple(float(v) for v in phantom.spacing_xyz))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    ox, oy, oz = phantom.origin_xyz
    img.SetOrigin((ox, oy, oz))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sitk.WriteImage(img, path)
    return path


# ---------------------------------------------------------------------------
# 5. HTML-галерея
# ---------------------------------------------------------------------------
def write_html_gallery(cases: Sequence[CaseRender],
                       out_html: str,
                       evaluations: Optional[Dict[str, EvaluationResult]] = None,
                       title: str = "Thickness autotest — visual report",
                       extra_files: Optional[Sequence[str]] = None) -> str:
    evaluations = evaluations or {}
    rows = []
    for case in cases:
        ev = evaluations.get(case.name)
        n_err = len(ev.errors) if ev else 0
        n_warn = len(ev.warns) if ev else 0
        status = ("ERROR" if n_err else ("WARN" if (n_warn or case.warnings) else "OK"))
        color = {"ERROR": "crimson", "WARN": "darkorange", "OK": "green"}[status]
        imgs = [p for p in (case.png, case.profile_png, case.metrics_png, case.nrrd) if p]
        cells = []
        for p in imgs:
            rel = os.path.relpath(p, os.path.dirname(out_html) or ".")
            if p.endswith(".png"):
                cells.append(f'<figure><img src="{html.escape(rel)}">'
                             f'<figcaption>{html.escape(os.path.basename(p))}</figcaption></figure>')
            else:
                cells.append(f'<li><a href="{html.escape(rel)}">{html.escape(os.path.basename(p))}</a> (NRRD)</li>')
        warn_list = "".join(f"<li>{html.escape(w)}</li>" for w in case.warnings)
        rows.append(
            f"<section class='case'><h3 style='color:{color}'>[{status}] "
            f"{html.escape(case.name)}</h3>"
            f"<div class='figs'>{''.join(cells)}</div>"
            f"<ul class='links'>{cells[-1] if case.nrrd else ''}</ul>"
            f"<details><summary>WARNINGS ({len(case.warnings)})</summary><ul>{warn_list}</ul></details>"
            f"</section>")
    extras = "".join(
        f"<figure><img src='{html.escape(os.path.relpath(p, os.path.dirname(out_html) or '.'))}'></figure>"
        for p in (extra_files or []) if p.endswith(".png"))
    doc = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:sans-serif;margin:1.5em;background:#fafafa}}
.case{{border:1px solid #ddd;background:#fff;padding:.6em 1em;margin-bottom:1.2em;border-radius:6px}}
.figs{{display:flex;flex-wrap:wrap;gap:.8em}}
figure{{margin:0}} img{{max-height:420px;border:1px solid #eee}}
figcaption{{font-size:.75em;color:#555}}
.links li{{font-size:.8em}} details summary{{cursor:pointer;color:#a33}}
</style><h1>{html.escape(title)}</h1>{''.join(rows)}<h2>Dashboard</h2>{extras}"""
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_html


# ---------------------------------------------------------------------------
# 6. Golden-сравнение картинок (детектор изменения пайплайна "глазами"+числом)
# ---------------------------------------------------------------------------
def compare_images(new_png: str, golden_png: str, threshold: float = 0.02) -> dict:
    """Численная оценка отличия двух PNG (средний |diff| по нормированным пикселям).

    Используется как "визуальный регрессионный тест": если после правки
    пайплайна картинка изменилась сильнее порога -- тест сообщает WARN и просит
    посмотреть галерею.  Разрешения могут различаться (менялся шаг решётки) --
    тогда берётся relative-diff по уменьшенной общей сетке.
    """
    from matplotlib.image import imread

    def load(p):
        a = imread(p)
        return a[..., :3].mean(axis=2) if a.ndim == 3 else a

    if not (os.path.exists(new_png) and os.path.exists(golden_png)):
        return {"available": False, "mean_abs_diff": float("nan"), "changed": False}
    a, b = load(new_png), load(golden_png)
    if a.shape != b.shape:
        # грубое усреднение до одинакового размера
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]; b = b[:h, :w]
    diff = np.abs(a - b)
    mad = float(diff.mean())
    return {"available": True, "mean_abs_diff": mad,
            "max_abs_diff": float(diff.max()),
            "frac_pixels_changed": float((diff > 0.05).mean()),
            "changed": bool(mad > threshold)}


# ---------------------------------------------------------------------------
# Мелкие утилиты
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(name))


def _write_sidecar(png_path: str, **info):
    """JSON рядом с PNG: какие срезы/предупреждения на картинке -- стабильность
    golden-файлов и трассируемость (видно, каким кодом получена картина)."""
    try:
        with open(os.path.splitext(png_path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    except OSError:
        pass


def align_to_phantom(field_arr: np.ndarray, phantom: Phantom,
                     shift: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Привести поле конвейера к решётке фантома (если конвейер режет bbox).

    Сейчас фантом генерируется на том же шаге, что и поле, поэтому достаточно
    сдвига; при изменении пайплайна (upscale/downscale) сюда добавится scale --
    см. Phantom.zones_for_field(scale=...).
    """
    from tests.analytics.phantom_generator import _crop_or_pad
    return _crop_or_pad(np.asarray(field_arr, dtype=np.float64), phantom.wall.shape, shift)
