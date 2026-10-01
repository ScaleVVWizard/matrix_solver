"""Окно истории решений."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QInputDialog,
    QMessageBox,
    QTableWidgetItem,
)

from database import Database, _pair_to_complex
from solver import METHOD_NAMES

UI_DIR = Path(__file__).parent


class HistoryWindow(QDialog):
    """Диалог с историей решений и возможностью загрузить матрицу обратно."""

    matrix_loaded = pyqtSignal(np.ndarray)

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_DIR / "history_window.ui"), self)

        self.db = db
        self._solution_ids: list[int] = []

        self.btnLoad.clicked.connect(self.on_load)
        self.btnDelete.clicked.connect(self.on_delete)
        self.btnRename.clicked.connect(self.on_rename)
        self.btnClose.clicked.connect(self.close)
        self.comboFilter.currentTextChanged.connect(self.refresh)
        self.editSearch.textChanged.connect(self.refresh)

        self.refresh()

    def refresh(self) -> None:
        """Перечитывает историю из БД с учётом фильтров."""
        rows = self.db.get_history(
            method_filter=self.comboFilter.currentText(),
            search=self.editSearch.text().strip() or None,
        )
        self.tableHistory.setRowCount(len(rows))
        self._solution_ids = []

        for i, row in enumerate(rows):
            (solution_id, _matrix_id, created_at, name, m_rows, m_cols,
             method, result_json) = row
            self._solution_ids.append(solution_id)

            raw_values = json.loads(result_json)
            values = [_pair_to_complex(v) for v in raw_values]
            preview_parts = []
            for idx, v in enumerate(values[:3]):
                if abs(v.imag) < 1e-12:
                    preview_parts.append(f"x{idx + 1}={v.real:.3g}")
                else:
                    sign = "+" if v.imag >= 0 else "-"
                    preview_parts.append(
                        f"x{idx + 1}={v.real:.3g}{sign}{abs(v.imag):.3g}i"
                    )
            preview = ", ".join(preview_parts)
            if len(values) > 3:
                preview += ", ..."

            size_text = f"{m_rows}×{m_cols}" if m_rows and m_cols else "—"

            self.tableHistory.setItem(i, 0, QTableWidgetItem(str(created_at)))
            self.tableHistory.setItem(i, 1, QTableWidgetItem(name or ""))
            self.tableHistory.setItem(i, 2, QTableWidgetItem(size_text))
            self.tableHistory.setItem(
                i, 3, QTableWidgetItem(METHOD_NAMES.get(method, method))
            )
            self.tableHistory.setItem(i, 4, QTableWidgetItem(preview))

    def _current_matrix_id(self) -> int | None:
        """Возвращает matrix_id для выделенной строки истории."""
        row = self.tableHistory.currentRow()
        if row < 0:
            return None
        solution_id = self._solution_ids[row]
        cursor = self.db.connection.cursor()
        cursor.execute(
            "SELECT matrix_id FROM solutions WHERE id = ?", (solution_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def on_load(self) -> None:
        """Загружает матрицу из выбранной записи и отправляет сигнал."""
        matrix_id = self._current_matrix_id()
        if matrix_id is None:
            QMessageBox.information(self, "Загрузка", "Выберите запись")
            return
        record = self.db.get_matrix(matrix_id)
        if not record:
            QMessageBox.warning(self, "Загрузка", "Матрица не найдена")
            return
        _, matrix = record
        self.matrix_loaded.emit(matrix)
        self.accept()

    def on_delete(self) -> None:
        """Удаляет выбранную запись."""
        row = self.tableHistory.currentRow()
        if row < 0:
            return
        name = self.tableHistory.item(row, 1).text()
        confirm = QMessageBox.question(
            self, "Удаление",
            f"Удалить запись «{name}»?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.db.delete_solution(self._solution_ids[row])
            self.refresh()

    def on_rename(self) -> None:
        """Переименовывает матрицу, привязанную к выбранной записи (UPDATE)."""
        row = self.tableHistory.currentRow()
        if row < 0:
            QMessageBox.information(self, "Переименование", "Выберите запись")
            return
        matrix_id = self._current_matrix_id()
        if matrix_id is None:
            QMessageBox.warning(self, "Переименование", "Матрица не найдена")
            return

        old_name = self.tableHistory.item(row, 1).text()
        new_name, ok = QInputDialog.getText(
            self, "Переименование",
            "Новое имя матрицы (во всех связанных записях):",
            text=old_name,
        )
        if ok and new_name.strip():
            self.db.rename_matrix(matrix_id, new_name.strip())
            self.refresh()