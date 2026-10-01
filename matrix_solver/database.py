"""
Модуль работы с базой данных SQLite.

Таблицы:
- matrices  — сохранённые матрицы (входные данные)
- solutions — история решений
- methods   — справочник методов (заполняется при первом запуске)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Путь к БД: рядом с .exe (при сборке) или рядом с модулем (при запуске из исходников)
if getattr(sys, "frozen", False):
    DB_FILE = Path(sys.executable).resolve().parent / "matrix_solver.db"
else:
    DB_FILE = Path(__file__).resolve().parent / "matrix_solver.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS matrices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    matrix_id INTEGER,
    method TEXT NOT NULL,
    result TEXT NOT NULL,
    steps TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (matrix_id) REFERENCES matrices(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT
);
"""

METHODS_SEED = [
    ("gauss", "Метод Гаусса", "Прямой метод с частичным выбором ведущего элемента"),
    ("cramer", "Метод Крамера", "Решение через определители (только для квадратных систем)"),
]


# ---------- Сериализация комплексных чисел ----------

def _complex_to_pair(value: complex) -> list:
    """Комплексное число -> [real, imag] (JSON-совместимо)."""
    return [float(value.real), float(value.imag)]


def _pair_to_complex(pair) -> complex:
    """[real, imag] -> complex."""
    return complex(pair[0], pair[1])


# ---------- Класс БД ----------

class Database:
    """Обёртка над SQLite для работы с матрицами и решениями."""

    def __init__(self, db_path: str | Path = DB_FILE):
        self.db_path = str(db_path)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self) -> None:
        """Создаёт таблицы и заполняет справочник методов."""
        cursor = self.connection.cursor()
        cursor.executescript(SCHEMA)
        cursor.executemany(
            "INSERT OR IGNORE INTO methods (code, name, description) VALUES (?, ?, ?)",
            METHODS_SEED,
        )
        self.connection.commit()

    # ---------- Матрицы ----------

    def save_matrix(self, name: str, matrix: np.ndarray) -> int:
        """Сохраняет матрицу, возвращает её id."""
        rows, cols = matrix.shape
        data = json.dumps(
            [[_complex_to_pair(complex(v)) for v in row] for row in matrix]
        )
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO matrices (name, rows, cols, data) VALUES (?, ?, ?, ?)",
            (name, rows, cols, data),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_matrix(self, matrix_id: int) -> Optional[Tuple[str, np.ndarray]]:
        """Возвращает (имя, матрица) по id или None."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT name, data FROM matrices WHERE id = ?", (matrix_id,))
        row = cursor.fetchone()
        if not row:
            return None
        name, data = row
        raw = json.loads(data)
        matrix = np.array(
            [[_pair_to_complex(v) for v in row_] for row_ in raw],
            dtype=np.complex128,
        )
        return name, matrix

    def matrix_name_exists(self, name: str) -> bool:
        """Проверяет, есть ли уже матрица с таким именем."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT 1 FROM matrices WHERE name = ? LIMIT 1", (name,))
        return cursor.fetchone() is not None

    def rename_matrix(self, matrix_id: int, new_name: str) -> None:
        """Переименовывает матрицу (UPDATE)."""
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE matrices SET name = ? WHERE id = ?",
            (new_name, matrix_id),
        )
        self.connection.commit()

    # ---------- Решения ----------

    def save_solution(
        self,
        matrix_id: int,
        method: str,
        result: np.ndarray,
        steps: str,
    ) -> int:
        """Сохраняет решение, возвращает его id."""
        result_json = json.dumps([_complex_to_pair(complex(v)) for v in result])
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO solutions (matrix_id, method, result, steps) VALUES (?, ?, ?, ?)",
            (matrix_id, method, result_json, steps),
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_history(
        self,
        method_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Tuple]:
        """
        Возвращает список записей истории:
        (solution_id, matrix_id, created_at, matrix_name, matrix_rows,
         matrix_cols, method, result_json)
        """
        query = """
            SELECT s.id, s.matrix_id, s.created_at, m.name, m.rows, m.cols,
                   s.method, s.result
            FROM solutions s
            LEFT JOIN matrices m ON s.matrix_id = m.id
            WHERE 1=1
        """
        params: List = []
        if method_filter and method_filter != "Все":
            method_map = {
                "Метод Гаусса": "gauss",
                "Метод Крамера": "cramer",
            }
            query += " AND s.method = ?"
            params.append(method_map.get(method_filter, method_filter))
        if search:
            query += " AND m.name LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY s.created_at DESC"

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def delete_solution(self, solution_id: int) -> None:
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM solutions WHERE id = ?", (solution_id,))
        self.connection.commit()

    def get_solution(self, solution_id: int) -> Optional[Tuple[str, str, str]]:
        """Возвращает (метод, результат_json, шаги) по id."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT method, result, steps FROM solutions WHERE id = ?",
            (solution_id,),
        )
        return cursor.fetchone()

    # ---------- Методы ----------

    def get_methods(self) -> List[Tuple[str, str]]:
        """Возвращает список (code, name) для выпадающего списка."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT code, name FROM methods ORDER BY id")
        return cursor.fetchall()

    def close(self) -> None:
        self.connection.close()