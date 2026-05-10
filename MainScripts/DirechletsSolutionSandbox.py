import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSlider, QGroupBox, QSpinBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from numba import jit


# ============================================
# Ручная сине-красная цветовая карта
# ============================================

def apply_blue_red_colormap(data):
    """
    Применяет сине-бело-красную цветовую карту вручную.
    0 → синий, 50 → белый, 100 → красный
    """
    # Нормализуем к [0, 1]
    normalized = np.clip(data / 100.0, 0, 1)

    height, width = normalized.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Векторизованное применение градиента
    # Синий (0,0,255) → Белый (255,255,255) → Красный (255,0,0)

    # Канал R: 0→255→255 (растёт до 50, потом постоянен)
    r = np.where(normalized < 0.5,
                 normalized * 2 * 255,
                 255)

    # Канал G: 0→255→0 (пик на 50)
    g = np.where(normalized < 0.5,
                 normalized * 2 * 255,
                 (1 - normalized) * 2 * 255)

    # Канал B: 255→255→0 (убывает после 50)
    b = np.where(normalized < 0.5,
                 255,
                 (1 - normalized) * 2 * 255)

    rgb[:, :, 0] = r.astype(np.uint8)
    rgb[:, :, 1] = g.astype(np.uint8)
    rgb[:, :, 2] = b.astype(np.uint8)

    return rgb


# ============================================
# Numba-ускоренный решатель
# ============================================

@jit(nopython=True, cache=True)
def laplace_solver(phi, mask, max_iter=1000, tol=1e-6):
    height, width = phi.shape

    # Фиксируем границы ПЕРЕД итерациями
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 2:
                phi[i, j] = 0.0
            elif mask[i, j] == 3:
                phi[i, j] = 100.0

    for iteration in range(max_iter):
        max_diff = 0.0
        phi_new = phi.copy()

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if mask[i, j] == 1:
                    count = 0
                    total = 0.0

                    if i > 0 and mask[i - 1, j] != 0:
                        total += phi[i - 1, j]
                        count += 1
                    if i < height - 1 and mask[i + 1, j] != 0:
                        total += phi[i + 1, j]
                        count += 1
                    if j > 0 and mask[i, j - 1] != 0:
                        total += phi[i, j - 1]
                        count += 1
                    if j < width - 1 and mask[i, j + 1] != 0:
                        total += phi[i, j + 1]
                        count += 1

                    if count > 0:
                        phi_new[i, j] = total / count
                        diff = abs(phi_new[i, j] - phi[i, j])
                        if diff > max_diff:
                            max_diff = diff

        phi[:] = phi_new[:]

        # Фиксируем границы ПОСЛЕ итерации
        for i in range(height):
            for j in range(width):
                if mask[i, j] == 2:
                    phi[i, j] = 0.0
                elif mask[i, j] == 3:
                    phi[i, j] = 100.0

        if max_diff < tol:
            return phi, iteration + 1, max_diff

    return phi, max_iter, max_diff


# ============================================
# Поток вычислений
# ============================================

class ComputeThread(QThread):
    progress = pyqtSignal(int, int, float)
    finished = pyqtSignal(np.ndarray, int, float)

    def __init__(self, phi, mask, max_iter=1000):
        super().__init__()
        self.phi = phi.copy()
        self.mask = mask.copy()
        self.max_iter = max_iter

    def run(self):
        phi_result, iterations, final_diff = laplace_solver(
            self.phi, self.mask, self.max_iter, tol=1e-6
        )
        self.finished.emit(phi_result, iterations, final_diff)


# ============================================
# Виджет холста
# ============================================

class CanvasWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(512, 512)
        self.setMaximumSize(600, 600)

        self.is_drawing = False
        self.brush_color = 0
        self.brush_size = 15
        self.mask_data = None
        self.canvas_size = 512

        self.colors = np.array([
            [0, 0, 0],  # 0: Чёрный
            [200, 200, 200],  # 1: Белый
            [0, 0, 255],  # 2: Синий
            [255, 0, 0],  # 3: Красный
        ], dtype=np.uint8)

        self.setMouseTracking(True)

    def setMaskData(self, mask):
        self.mask_data = mask
        self.canvas_size = mask.shape[0]
        self.update_image()

    def setBrushColor(self, color):
        self.brush_color = color

    def setBrushSize(self, size):
        self.brush_size = size

    def update_image(self):
        if self.mask_data is None:
            return

        rgb = self.colors[self.mask_data]
        rgb_flat = rgb.reshape(-1, 3)

        self.qimage = QImage(
            rgb_flat.tobytes(),
            self.canvas_size,
            self.canvas_size,
            self.canvas_size * 3,
            QImage.Format_RGB888
        ).copy()

        self.update()

    def get_mask_coords(self, pos):
        x = int(pos.x() * self.canvas_size / self.width())
        y = int(pos.y() * self.canvas_size / self.height())
        return np.clip(x, 0, self.canvas_size - 1), np.clip(y, 0, self.canvas_size - 1)

    def draw(self, pos):
        if self.mask_data is None:
            return

        x, y = self.get_mask_coords(pos)
        h, w = self.mask_data.shape
        r = self.brush_size

        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2

        self.mask_data[y0:y1, x0:x1][circle] = self.brush_color
        self.update_image()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        pixmap = QPixmap.fromImage(self.qimage)
        pixmap = pixmap.scaled(self.width(), self.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.draw(event.pos())
        elif event.button() == Qt.RightButton:
            self.is_drawing = False

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.draw(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False


# ============================================
# Основное приложение
# ============================================

class LaplaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Решатель уравнения Лапласа")
        self.setGeometry(100, 100, 1400, 900)

        self.canvas_size = 512
        self.brush_size = 15

        self.mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        self.phi = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float64)

        self.init_ui()
        self.reset_canvas()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель
        left_panel = QVBoxLayout()

        self.canvas_widget = CanvasWidget()
        self.canvas_widget.setMaskData(self.mask)
        left_panel.addWidget(self.canvas_widget)

        # Результат с эквипотенциальными линиями
        result_group = QGroupBox("Результат (Φ)")
        result_layout = QVBoxLayout()

        self.result_widget = pg.GraphicsLayoutWidget()
        self.result_widget.setFixedSize(600, 300)

        self.result_plot = self.result_widget.addPlot()
        self.result_plot.hideAxis('bottom')
        self.result_plot.hideAxis('left')
        self.result_plot.setAspectLocked(lock=True, ratio=1)

        self.result_image = pg.ImageItem()
        self.result_plot.addItem(self.result_image)

        self.contour_lines = []

        result_layout.addWidget(self.result_widget)
        result_group.setLayout(result_layout)
        left_panel.addWidget(result_group)

        main_layout.addLayout(left_panel)

        # Правая панель
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        brush_group = QGroupBox("Кисти")
        brush_layout = QVBoxLayout()

        self.brush_combo = QComboBox()
        self.brush_combo.addItems(['Чёрная (фон)', 'Белая (маска)',
                                   'Синяя (Φ=0)', 'Красная (Φ=100)'])
        self.brush_combo.currentIndexChanged.connect(
            lambda i: self.canvas_widget.setBrushColor(i))
        brush_layout.addWidget(QLabel("Кисть:"))
        brush_layout.addWidget(self.brush_combo)

        brush_layout.addWidget(QLabel("Размер:"))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(
            lambda v: self.canvas_widget.setBrushSize(v))
        brush_layout.addWidget(self.brush_size_slider)

        brush_group.setLayout(brush_layout)
        right_panel.addWidget(brush_group)

        control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout()

        self.reset_btn = QPushButton("Очистить")
        self.reset_btn.clicked.connect(self.reset_canvas)
        control_layout.addWidget(self.reset_btn)

        self.solve_btn = QPushButton("Запустить расчёт")
        self.solve_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.solve_btn.clicked.connect(self.start_computation)
        control_layout.addWidget(self.solve_btn)

        self.contour_btn = QPushButton("Показать линии (10 шт)")
        self.contour_btn.clicked.connect(self.toggle_contours)
        self.contour_btn.setCheckable(True)
        self.contour_btn.setChecked(True)
        control_layout.addWidget(self.contour_btn)

        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)

        params_group = QGroupBox("Параметры")
        params_layout = QVBoxLayout()

        params_layout.addWidget(QLabel("Итераций:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setMinimum(100)
        self.max_iter_spin.setMaximum(10000)
        self.max_iter_spin.setValue(2000)
        params_layout.addWidget(self.max_iter_spin)

        params_layout.addWidget(QLabel("Линий эквипотенциалей:"))
        self.contour_count_spin = QSpinBox()
        self.contour_count_spin.setMinimum(0)
        self.contour_count_spin.setMaximum(50)
        self.contour_count_spin.setValue(10)
        params_layout.addWidget(self.contour_count_spin)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        params_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: orange; font-family: monospace;")
        params_layout.addWidget(self.progress_label)

        self.status_label = QLabel("Готов")
        self.status_label.setStyleSheet("color: gray;")
        params_layout.addWidget(self.status_label)

        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: blue; font-weight: bold; font-family: monospace;")
        params_layout.addWidget(self.stats_label)

        params_group.setLayout(params_layout)
        right_panel.addWidget(params_group)

        info = QGroupBox("Инфо")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(
            "🖤 Чёрная: фон (игнор)\n⬜ Белая: решение\n"
            "🔵 Синяя: Φ=0 (фикс)\n🔴 Красная: Φ=100 (фикс)\n\n"
            "📊 Цвета результата:\n"
            "🔵 Синий = 0\n"
            "⬜ Белый = 50\n"
            "🔴 Красный = 100"
        ))
        info.setLayout(info_layout)
        right_panel.addWidget(info)

        right_panel.addStretch()
        main_layout.addLayout(right_panel)

    def reset_canvas(self):
        self.mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        self.phi = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float64)
        self.canvas_widget.setMaskData(self.mask)

        self.result_image.setImage(np.zeros((self.canvas_size, self.canvas_size)))
        self.clear_contours()

        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.status_label.setText("Готов")
        self.stats_label.setText("")

    def clear_contours(self):
        for line in self.contour_lines:
            self.result_plot.removeItem(line)
        self.contour_lines = []

    def draw_contours(self, phi_data, levels=10):
        self.clear_contours()

        if levels <= 0:
            return

        contour_values = np.linspace(10, 90, levels)

        for level in contour_values:
            try:
                from pyqtgraph import IsocurveItem
                curve = IsocurveItem(level=level, pen=pg.mkPen(color='w', width=1, style=Qt.DashLine))
                curve.setData(phi_data)
                self.result_plot.addItem(curve)
                self.contour_lines.append(curve)
            except Exception as e:
                print(f"Контур {level}: {e}")

    def toggle_contours(self):
        if self.contour_btn.isChecked():
            if self.phi is not None and np.any(self.phi != 0):
                self.draw_contours(self.phi, self.contour_count_spin.value())
        else:
            self.clear_contours()

    def prepare_data(self):
        phi = np.full((self.canvas_size, self.canvas_size), np.nan, dtype=np.float64)
        mask = self.mask.copy()

        phi[self.mask == 1] = 50.0
        phi[self.mask == 2] = 0.0
        phi[self.mask == 3] = 100.0

        return phi, mask

    def start_computation(self):
        self.status_label.setText("Вычисление...")
        self.solve_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.clear_contours()

        phi, mask = self.prepare_data()

        if np.sum(mask == 2) == 0:
            self.status_label.setText("⚠️ Нужна синяя граница (Φ=0)!")
            self.solve_btn.setEnabled(True)
            return
        if np.sum(mask == 3) == 0:
            self.status_label.setText("⚠️ Нужна красная граница (Φ=100)!")
            self.solve_btn.setEnabled(True)
            return
        if np.sum(mask == 1) == 0:
            self.status_label.setText("⚠️ Нужна белая область!")
            self.solve_btn.setEnabled(True)
            return

        self.compute_thread = ComputeThread(phi, mask, self.max_iter_spin.value())
        self.compute_thread.finished.connect(self.on_finished)
        self.compute_thread.start()

    def on_finished(self, phi_result, iterations, final_diff):
        self.phi = phi_result
        self.solve_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText("✅ Готово!")

        display_phi = np.nan_to_num(phi_result, nan=0.0)

        # ✅ ИСПРАВЛЕНО: ручная цветовая карта без pyqtgraph ColorMap
        rgb_image = apply_blue_red_colormap(display_phi)

        self.result_image.setImage(rgb_image, levels=(0, 255))

        if self.contour_btn.isChecked():
            self.draw_contours(display_phi, self.contour_count_spin.value())

        white = (self.mask == 1)
        if np.sum(white) > 0:
            vals = phi_result[white]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                min_v, max_v = np.min(vals), np.max(vals)

                warning = ""
                if min_v < -0.01 or max_v > 100.01:
                    warning = "\n⚠️ Выход за [0,100]!"

                self.stats_label.setText(
                    f"Итераций: {iterations}\n"
                    f"Min Φ: {min_v:10.4f}\n"
                    f"Max Φ: {max_v:10.4f}\n"
                    f"Δ финал: {final_diff:.2e}{warning}"
                )

    def closeEvent(self, event):
        if hasattr(self, 'compute_thread') and self.compute_thread.isRunning():
            self.compute_thread.terminate()
            self.compute_thread.wait()
        event.accept()


# ============================================
# Запуск
# ============================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')

    window = LaplaceApp()
    window.show()
    sys.exit(app.exec_())