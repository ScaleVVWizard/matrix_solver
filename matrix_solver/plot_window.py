"""Окно графической интерпретации решений (только 2D для 2×2, 3D для 3×3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox, QVBoxLayout

UI_DIR = Path(__file__).parent


class PlotWindow(QDialog):
    """Диалог с графиком. Тип определяется размером системы."""

    def __init__(self, matrix: np.ndarray, b: np.ndarray,
                 solution: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_DIR / "plot_window.ui"), self)

        self.matrix = matrix
        self.b = b
        self.solution = solution

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self.plotContainer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.btnSavePng.clicked.connect(self.save_png)
        self.btnClose.clicked.connect(self.close)

        self.build()

    def build(self) -> None:
        """Строит график в зависимости от размера системы."""
        self.figure.clear()
        n = self.matrix.shape[0]

        try:
            if n == 2:
                self._plot_2d()
            elif n == 3:
                self._plot_3d()
            else:
                ax = self.figure.add_subplot(111)
                ax.text(
                    0.5, 0.5,
                    "Графики доступны только для систем 2×2 и 3×3",
                    ha="center", va="center", transform=ax.transAxes,
                )
                ax.axis("off")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка построения", str(exc))
            return

        self.canvas.draw()

    # ---------- Типы графиков ----------

    def _plot_2d(self) -> None:
        """Прямые для системы 2×2 + точка решения."""
        ax = self.figure.add_subplot(111)
        n = self.matrix.shape[0]
        if n < 2:
            ax.text(0.5, 0.5, "Нужно минимум 2 уравнения", ha="center")
            return

        sol = [complex(v).real for v in self.solution]

        # Центр обзора — точка решения (если есть), иначе начало координат
        if len(sol) >= 2:
            cx, cy = sol[0], sol[1]
        else:
            cx, cy = 0.0, 0.0

        span = max(5.0, abs(cx) * 1.5, abs(cy) * 1.5)
        x_vals = np.linspace(cx - span, cx + span, 300)

        for i in range(min(n, 2)):
            a1 = complex(self.matrix[i, 0]).real
            a2 = complex(self.matrix[i, 1]).real
            b = complex(self.b[i]).real

            if abs(a2) < 1e-12:
                if abs(a1) > 1e-12:
                    ax.axvline(b / a1, label=f"Ур. {i + 1}")
            else:
                y_vals = (b - a1 * x_vals) / a2
                ax.plot(x_vals, y_vals, label=f"Ур. {i + 1}")

        if len(sol) >= 2:
            ax.plot(sol[0], sol[1], "ro", markersize=10, label="Решение")
            ax.annotate(
                f"({sol[0]:.3g}; {sol[1]:.3g})",
                (sol[0], sol[1]),
                textcoords="offset points", xytext=(10, 10),
            )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_title(
            f"Графическая интерпретация (2D)\n"
            f"Решение: x1={sol[0]:.3g}, x2={sol[1]:.3g}"
            if len(sol) >= 2 else "Графическая интерпретация (2D)"
        )

    def _plot_3d(self) -> None:
        """Плоскости для системы 3×3 + точка решения (в одном масштабе)."""
        ax = self.figure.add_subplot(111, projection="3d")
        n = self.matrix.shape[0]
        if n < 3:
            ax.text2D(0.5, 0.5, "Нужно 3 уравнения",
                      transform=ax.transAxes, ha="center")
            return

        sol = [complex(v).real for v in self.solution]

        if len(sol) >= 3:
            cx, cy, cz = sol[0], sol[1], sol[2]
        else:
            cx, cy, cz = 0.0, 0.0, 0.0

        span = max(5.0, abs(cx) * 1.5, abs(cy) * 1.5, abs(cz) * 1.5)

        x = np.linspace(cx - span, cx + span, 25)
        y = np.linspace(cy - span, cy + span, 25)
        X, Y = np.meshgrid(x, y)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i in range(min(n, 3)):
            a1 = complex(self.matrix[i, 0]).real
            a2 = complex(self.matrix[i, 1]).real
            a3 = complex(self.matrix[i, 2]).real
            b = complex(self.b[i]).real

            if abs(a3) < 1e-12:
                if abs(a1) > 1e-12:
                    x0 = b / a1
                    Yv, Zv = np.meshgrid(
                        y, np.linspace(cz - span, cz + span, 15)
                    )
                    Xv = np.full_like(Yv, x0)
                    ax.plot_surface(
                        Xv, Yv, Zv, alpha=0.35, color=colors[i % 3]
                    )
                continue

            Z = (b - a1 * X - a2 * Y) / a3
            ax.plot_surface(
                X, Y, Z, alpha=0.35,
                color=colors[i % 3],
                edgecolor="none",
            )

        if len(sol) >= 3:
            ax.scatter(
                sol[0], sol[1], sol[2],
                color="red", s=120, depthshade=False,
                zorder=10,
            )
            ax.text(
                sol[0], sol[1], sol[2],
                f"  ({sol[0]:.3g}; {sol[1]:.3g}; {sol[2]:.3g})",
                color="red", fontsize=10,
            )

        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title(
            f"Пересечение плоскостей (3D)\n"
            f"Решение: x1={sol[0]:.3g}, x2={sol[1]:.3g}, x3={sol[2]:.3g}"
            if len(sol) >= 3 else "Пересечение плоскостей (3D)"
        )

    # ---------- Сохранение ----------

    def save_png(self) -> None:
        """Сохраняет текущий график в PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "plot.png", "PNG (*.png)"
        )
        if path:
            self.figure.savefig(path, dpi=150)
            QMessageBox.information(
                self, "Сохранение", f"График сохранён:\n{path}"
            )