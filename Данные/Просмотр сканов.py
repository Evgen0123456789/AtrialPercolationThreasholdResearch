# viewer.py
import sys
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


def read_nrrd(path):
    """Читает NRRD файл (заголовок + сырые данные)."""
    sizes = None
    endian = 'little'
    dtype = np.float32

    with open(path, 'rb') as f:
        # Читаем заголовок построчно
        while True:
            line = f.readline()
            if line == b'\n' or line == b'':  # Конец заголовка
                break
            line = line.decode('ascii', errors='ignore').strip()
            if line.startswith('sizes:'):
                sizes = tuple(map(int, line.split(':')[1].split()))
            elif line.startswith('endian:'):
                endian = line.split(':')[1].strip()
            elif line.startswith('type:'):
                t = line.split(':')[1].strip()
                dtype = {'float': np.float32, 'double': np.float64,
                         'int': np.int32, 'uint8': np.uint8}.get(t, np.float32)

        # Читаем бинарные данные
        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape(sizes, order='F')  # NRRD = Fortran order (Z,Y,X)

        if endian == 'big':
            data = data.byteswap()

    return data


app = QtWidgets.QApplication(sys.argv)
files = sys.argv[1:] if len(sys.argv) > 1 else ["U_fine.nrrd", "AWT_fine.nrrd"]

for path in files:
    try:
        data = read_nrrd(path)
        win = pg.ImageView()
        for i in range(data.shape[0]):
            win.setImage(data[i])
            time.sleep(0.01)
        win.show()
        win.setWindowTitle(f"{path} | {data.shape}")
        print(f"[Loaded] {path} → {data.shape}")

    except Exception as e:
        print(f"[Error] {path}: {e}")

print("[Viewer] Scroll: slices | Right-click: pan/zoom | Close: exit")
sys.exit(app.exec_())