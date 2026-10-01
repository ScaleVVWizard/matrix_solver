"""
Главный модуль приложения «Матричный калькулятор».

Запуск: python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import QRegularExpression, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QRegularExpressionValidator,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplashScreen,
    QStyledItemDelegate,
    QTableWidgetItem,
)

from database import Database
from solver import (
    METHOD_CRAMER,
    METHOD_GAUSS,
    cramer_solve,
    format_complex,
    format_matrix,
    gauss_solve,
    matrix_determinant,
    matrix_inverse,
    parse_cell,
    parse_matrix_from_table,
)

UI_DIR = Path(__file__).parent

# Разрешённые символы при вводе в таблицу
ALLOWED_CHARS = set("0123456789+-.,i")
# Цвета подсветки
COLOR_VALID = QColor("white")
COLOR_INVALID = QColor("#ffcccc")   # светло-красный
COLOR_TOO_BIG = QColor("#ffe9b3")   # светло-жёлтый
# Максимальная длина вводимой строки в ячейке
MAX_CELL_LEN = 15


class CellDelegate(QStyledItemDelegate):
    """
    Делегат для ячеек таблицы:
    - ставит QRegularExpressionValidator на редактор ячейки
      (нужно для случая двойного клика, когда открывается встроенный QLineEdit);
    - ограничивает длину строки;
    - подсвечивает невалидный ввод после закрытия редактора.
    """

    # Регексп: разрешены цифры, знаки, точка, запятая, буква i
    # Пример: -12.5+3.14i, i, -i, 100, .5
    PATTERN = QRegularExpression(r"^[0-9+\-.,i]*$")

    def createEditor(self, parent, option, index):  # noqa: N802
        editor = QLineEdit(parent)
        editor.setMaxLength(MAX_CELL_LEN)
        validator = QRegularExpressionValidator(self.PATTERN, editor)
        editor.setValidator(validator)
        return editor


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(str(UI_DIR / "main_window.ui"), self)

        self.db = Database()
        self.current_matrix_id: int | None = None
        self.current_matrix: np.ndarray | None = None
        self.current_b: np.ndarray | None = None
        self.current_solution: np.ndarray | None = None

        # Делегат с валидатором — вешаем на таблицу
        self._cell_delegate = CellDelegate(self.tableMatrix)
        self.tableMatrix.setItemDelegate(self._cell_delegate)

        self._setup_branding()
        self._setup_shortcuts()
        self._setup_context_menu()
        self._connect_signals()
        self._resize_table()

        self.tableMatrix.installEventFilter(self)
        self.tableMatrix.itemChanged.connect(self._on_item_changed)

        self.statusbar.showMessage("Готово")

    # ---------- Фильтр ввода ----------

    def eventFilter(self, obj, event):  # noqa: N802
        """Блокирует ввод символов, не входящих в ALLOWED_CHARS (быстрый ввод)."""
        if obj is self.tableMatrix and event.type() == event.Type.KeyPress:
            text = event.text()
            key = event.key()

            if key in (
                Qt.Key.Key_Backspace,
                Qt.Key.Key_Delete,
                Qt.Key.Key_Left,
                Qt.Key.Key_Right,
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
                Qt.Key.Key_Home,
                Qt.Key.Key_End,
                Qt.Key.Key_Tab,
                Qt.Key.Key_Return,
                Qt.Key.Key_Enter,
                Qt.Key.Key_Escape,
            ):
                return False

            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                return False

            if text and text not in ALLOWED_CHARS:
                return True

        return super().eventFilter(obj, event)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Подсвечивает ячейку в зависимости от корректности числа."""
        text = item.text().strip()
        if not text:
            item.setBackground(COLOR_VALID)
            item.setToolTip("")
            return

        try:
            parse_cell(text)
        except ValueError as exc:
            msg = str(exc)
            if "слишком большое" in msg or "слишком большая" in msg:
                item.setBackground(COLOR_TOO_BIG)
            else:
                item.setBackground(COLOR_INVALID)
            item.setToolTip(msg)
            return

        item.setBackground(COLOR_VALID)
        item.setToolTip("")

    # ---------- Настройка ----------

    def _setup_branding(self) -> None:
        logo_path = UI_DIR / "logo.png"
        if not logo_path.exists():
            return
        self.setWindowIcon(QIcon(str(logo_path)))
        pixmap = QPixmap(str(logo_path))
        if hasattr(self, "labelLogo"):
            self.labelLogo.setPixmap(
                pixmap.scaled(
                    90, 90,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.on_load)
        QShortcut(QKeySequence("Delete"), self.tableMatrix,
                  activated=self.on_clear_cell)

    def _setup_context_menu(self) -> None:
        self.tableMatrix.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tableMatrix.customContextMenuRequested.connect(self.on_table_context_menu)

    def _connect_signals(self) -> None:
        self.spinRows.valueChanged.connect(self._resize_table)
        self.spinCols.valueChanged.connect(self._resize_table)
        self.btnSolve.clicked.connect(self.on_solve)
        self.btnClear.clicked.connect(self.on_clear)
        self.btnSave.clicked.connect(self.on_save)
        self.btnHistory.clicked.connect(self.on_history)
        self.btnPlot.clicked.connect(self.on_plot)
        self.btnInverse.clicked.connect(self.on_inverse)
        self.btnDeterminant.clicked.connect(self.on_determinant)
        self.btnExit.clicked.connect(self.close)
        self.actionLoad.triggered.connect(self.on_load)
        self.actionSave.triggered.connect(self.on_save)
        self.actionExit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.on_about)

    def _resize_table(self) -> None:
        rows = self.spinRows.value()
        cols = self.spinCols.value() + 1
        self.tableMatrix.setRowCount(rows)
        self.tableMatrix.setColumnCount(cols)

        headers = [f"x{i + 1}" for i in range(cols - 1)] + ["b"]
        self.tableMatrix.setHorizontalHeaderLabels(headers)

    # ---------- Чтение ввода ----------

    def _read_matrix_a_and_b(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            table_data = self._read_table()
            A_full = parse_matrix_from_table(
                table_data, self.checkComplex.isChecked()
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка ввода", str(exc))
            return None

        n = A_full.shape[0]
        if A_full.shape[1] != n + 1:
            QMessageBox.warning(
                self, "Ошибка",
                f"Таблица должна содержать {n} столбцов матрицы A "
                f"и один столбец вектора b.\n"
                f"Сейчас столбцов: {A_full.shape[1]}, ожидается: {n + 1}."
            )
            return None

        A = A_full[:, :n]
        b = A_full[:, n]
        return A, b

    def _read_matrix_a(self) -> np.ndarray | None:
        parsed = self._read_matrix_a_and_b()
        if parsed is None:
            return None
        A, _b = parsed
        return A

    # ---------- Обработчики ----------

    def on_solve(self) -> None:
        parsed = self._read_matrix_a_and_b()
        if parsed is None:
            return
        A, b = parsed

        method_name = self.comboMethod.currentText()
        verbose = self.checkSteps.isChecked()

        self.progressBar.setValue(10)
        self.statusbar.showMessage(f"Решение методом: {method_name}...")

        try:
            if method_name == "Метод Гаусса":
                solution, steps = gauss_solve(A, b, verbose=verbose)
                method_code = METHOD_GAUSS
            else:
                solution, steps = cramer_solve(A, b)
                method_code = METHOD_CRAMER
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка вычислений", str(exc))
            self.progressBar.setValue(0)
            return

        self.progressBar.setValue(100)

        if solution is None:
            self.textResult.setPlainText(f"Решение не найдено.\n\n{steps}")
            self.statusbar.showMessage("Решение не найдено")
            return

        self.current_matrix = A
        self.current_b = b
        self.current_solution = solution

        output = "Решение:\n"
        for i, value in enumerate(solution):
            output += f"  x{i + 1} = {format_complex(complex(value))}\n"
        output += "\n"
        output += steps
        self.textResult.setPlainText(output)

        self.statusbar.showMessage(
            f"Готово. Метод: {method_name}. "
            "Сохраните результат кнопкой «Сохранить»."
        )

    def on_inverse(self) -> None:
        A = self._read_matrix_a()
        if A is None:
            return
        try:
            inv_A = matrix_inverse(A)
        except ValueError as exc:
            QMessageBox.warning(self, "Обратная матрица", str(exc))
            return

        output = "Обратная матрица A⁻¹:\n\n"
        output += format_matrix(inv_A)
        output += "\n"
        self.textResult.setPlainText(output)
        self.statusbar.showMessage("Обратная матрица построена")

    def on_determinant(self) -> None:
        A = self._read_matrix_a()
        if A is None:
            return
        try:
            det = matrix_determinant(A)
        except ValueError as exc:
            QMessageBox.warning(self, "Определитель", str(exc))
            return

        output = (
            f"Определитель матрицы A:\n\n"
            f"  det(A) = {format_complex(det)}\n"
        )
        self.textResult.setPlainText(output)
        self.statusbar.showMessage("Определитель вычислен")

    def on_clear(self) -> None:
        self.tableMatrix.clearContents()
        self.tableMatrix.setRowCount(self.spinRows.value())
        self.tableMatrix.setColumnCount(self.spinCols.value() + 1)
        self.textResult.clear()
        self.progressBar.setValue(0)
        self.current_matrix_id = None
        self.current_matrix = None
        self.current_b = None
        self.current_solution = None
        self.statusbar.showMessage("Очищено")

    def on_save(self) -> None:
        A = self._read_matrix_a()
        if A is None:
            return

        name, ok = QInputDialog.getText(self, "Сохранение", "Имя матрицы:")
        if not ok or not name.strip():
            return
        name = name.strip()

        if len(name) > 100:
            QMessageBox.warning(
                self, "Сохранение",
                "Имя слишком длинное (максимум 100 символов)."
            )
            return

        if self.db.matrix_name_exists(name):
            confirm = QMessageBox.question(
                self, "Сохранение",
                f"Матрица с именем «{name}» уже есть в истории.\n"
                "Сохранить ещё одну с таким же именем?"
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        b = self.current_b
        if b is not None and self.current_solution is not None:
            full = np.hstack([A, b.reshape(-1, 1)])
        else:
            full = A

        matrix_id = self.db.save_matrix(name, full)

        if self.current_solution is not None and b is not None:
            method_map = {
                "Метод Гаусса": METHOD_GAUSS,
                "Метод Крамера": METHOD_CRAMER,
            }
            method_code = method_map.get(
                self.comboMethod.currentText(), METHOD_GAUSS
            )
            self.db.save_solution(
                matrix_id, method_code, self.current_solution,
                self.textResult.toPlainText()
            )

        self.current_matrix_id = matrix_id
        self.statusbar.showMessage(f"Сохранено: {name}")

    def on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить матрицу", "", "CSV (*.csv);;Все файлы (*)"
        )
        if not path:
            return
        try:
            data = np.loadtxt(path, delimiter=",", dtype=str)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[0] > 20 or data.shape[1] > 21:
                QMessageBox.warning(
                    self, "Загрузка",
                    "Матрица слишком большая (максимум 20×20)."
                )
                return

            rows = data.shape[0]
            cols = data.shape[1] - 1
            if rows < 2 or cols < 2:
                QMessageBox.warning(
                    self, "Загрузка", "Минимальный размер — 2×2."
                )
                return
            if rows > self.spinRows.maximum():
                QMessageBox.warning(
                    self, "Загрузка",
                    f"Слишком много строк (максимум {self.spinRows.maximum()})."
                )
                return
            if cols > self.spinCols.maximum():
                QMessageBox.warning(
                    self, "Загрузка",
                    f"Слишком много столбцов (максимум {self.spinCols.maximum()})."
                )
                return

            self.spinRows.setValue(rows)
            self.spinCols.setValue(cols)
            self._resize_table()
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self.tableMatrix.setItem(
                        i, j, QTableWidgetItem(str(data[i, j]))
                    )
            self.statusbar.showMessage(f"Загружено: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))

    def on_history(self) -> None:
        from history_window import HistoryWindow
        self.history_window = HistoryWindow(self.db, self)
        self.history_window.matrix_loaded.connect(self._on_matrix_loaded)
        self.history_window.show()

    def on_plot(self) -> None:
        parsed = self._read_matrix_a_and_b()
        if parsed is None:
            return
        A, b = parsed
        n = A.shape[0]

        if n not in (2, 3):
            QMessageBox.information(
                self, "Графики",
                "Графики доступны только для систем 2×2 и 3×3."
            )
            return

        if self.checkComplex.isChecked():
            QMessageBox.information(
                self, "Графики",
                "Графики недоступны для комплексных чисел."
            )
            return

        if self.current_solution is None:
            QMessageBox.information(
                self, "Графики",
                "Сначала решите систему кнопкой «Решить»."
            )
            return

        from plot_window import PlotWindow
        self.plot_window = PlotWindow(A, b, self.current_solution, self)
        self.plot_window.show()

    def on_about(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            "Матричный калькулятор\n\n"
            "Семестровая работа студента группы КБ-241.\n\n"
            "Реализованы методы: Гаусс, Крамер.\n"
            "Поддержка комплексных чисел, пошаговый вывод, графики,\n"
            "операции с матрицами (обратная, определитель).",
        )

    def on_clear_cell(self) -> None:
        for item in self.tableMatrix.selectedItems():
            item.setText("")

    def on_table_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_clear = menu.addAction("Очистить ячейку")
        act_copy = menu.addAction("Копировать")
        act_paste = menu.addAction("Вставить")
        action = menu.exec(self.tableMatrix.viewport().mapToGlobal(pos))
        if action == act_clear:
            self.on_clear_cell()
        elif action == act_copy:
            items = self.tableMatrix.selectedItems()
            if items:
                QApplication.clipboard().setText(items[0].text())
        elif action == act_paste:
            items = self.tableMatrix.selectedItems()
            if items:
                items[0].setText(QApplication.clipboard().text())

    def _on_matrix_loaded(self, matrix: np.ndarray) -> None:
        rows, cols_total = matrix.shape
        self.spinRows.setValue(rows)
        self.spinCols.setValue(cols_total - 1 if cols_total > 1 else 1)
        self._resize_table()
        for i in range(rows):
            for j in range(cols_total):
                value = complex(matrix[i, j])
                text = format_complex(value)
                self.tableMatrix.setItem(i, j, QTableWidgetItem(text))
        self.statusbar.showMessage("Матрица загружена из истории")

    def _read_table(self) -> list[list[str]]:
        rows = self.tableMatrix.rowCount()
        cols = self.tableMatrix.columnCount()
        data: list[list[str]] = []
        for i in range(rows):
            row: list[str] = []
            for j in range(cols):
                item = self.tableMatrix.item(i, j)
                row.append(item.text() if item else "0")
            data.append(row)
        return data

    def closeEvent(self, event) -> None:  # noqa: N802
        self.db.close()
        super().closeEvent(event)


# ---------- Сплэш ----------

def _make_splash() -> QSplashScreen:
    """Сплэш ровно по размеру logo.png (без серых полей)."""
    logo_path = UI_DIR / "logo.png"
    if logo_path.exists():
        pixmap = QPixmap(str(logo_path))
        return QSplashScreen(pixmap)

    pixmap = QPixmap(770, 770)
    pixmap.fill(QColor("#000000"))

    painter = QPainter(pixmap)
    painter.setPen(QColor("white"))

    font = painter.font()
    font.setPointSize(28)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(
        pixmap.rect(),
        Qt.AlignmentFlag.AlignCenter,
        "Матричный\nкалькулятор",
    )

    font.setPointSize(14)
    font.setBold(False)
    painter.setFont(font)
    painter.drawText(
        pixmap.rect().adjusted(0, 120, 0, 0),
        Qt.AlignmentFlag.AlignCenter,
        "Семестровая работа · КБ-241",
    )
    painter.end()
    return QSplashScreen(pixmap)


def main() -> None:
    app = QApplication(sys.argv)

    splash = _make_splash()
    splash.show()
    app.processEvents()

    window = MainWindow()

    QTimer.singleShot(1500, lambda: (window.show(), splash.finish(window)))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()