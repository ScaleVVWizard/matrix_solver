"""
Математическое ядро приложения: решение СЛАУ и операции с матрицами.

Поддерживает:
- Метод Гаусса с частичным выбором ведущего элемента (с пошаговым выводом)
- Метод Крамера
- Определитель, обратная матрица
- Комплексные числа (через numpy.complex128)
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

# Константы — коды методов
METHOD_GAUSS = "gauss"
METHOD_CRAMER = "cramer"

# Человекочитаемые названия методов
METHOD_NAMES = {
    METHOD_GAUSS: "Метод Гаусса",
    METHOD_CRAMER: "Метод Крамера",
}

# Порог для определения нуля
EPS_ZERO = 1e-12

# Максимальное абсолютное значение при вводе числа
MAX_INPUT_ABS = 1_000_000  # 10^6


# ---------- Парсинг ввода ----------

def parse_cell(text: str) -> complex:
    """
    Парсит строку в комплексное число.

    Поддерживает: '5', '-3.2', '2+3i', '2-3i', '-i', 'i', '3i', '2.5+0.1i'.
    При некорректном вводе или превышении MAX_INPUT_ABS возбуждает ValueError.
    """
    text = text.strip().replace(" ", "").replace(",", ".")
    if not text:
        return complex(0, 0)

    if "i" not in text:
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"Некорректное число: {text}") from exc
        if abs(value) > MAX_INPUT_ABS:
            raise ValueError(
                f"Число слишком большое по модулю "
                f"(максимум {MAX_INPUT_ABS:g}): {text}"
            )
        return complex(value, 0)

    # Есть 'i' — разбираем вручную
    if text in ("i", "+i"):
        return complex(0, 1)
    if text == "-i":
        return complex(0, -1)

    body = text.rstrip("i")
    split_pos = -1
    for i in range(1, len(body)):
        if body[i] in "+-":
            split_pos = i

    if split_pos == -1:
        try:
            imag = float(body)
        except ValueError as exc:
            raise ValueError(f"Некорректное комплексное число: {text}") from exc
        if abs(imag) > MAX_INPUT_ABS:
            raise ValueError(
                f"Мнимая часть слишком большая: {text}"
            )
        return complex(0, imag)

    real_part = body[:split_pos]
    imag_part = body[split_pos:]

    try:
        real = float(real_part) if real_part else 0.0
    except ValueError as exc:
        raise ValueError(f"Некорректная вещественная часть: {text}") from exc

    if imag_part in ("+", ""):
        imag = 1.0
    elif imag_part == "-":
        imag = -1.0
    else:
        try:
            imag = float(imag_part)
        except ValueError as exc:
            raise ValueError(f"Некорректная мнимая часть: {text}") from exc

    if abs(real) > MAX_INPUT_ABS or abs(imag) > MAX_INPUT_ABS:
        raise ValueError(
            f"Число слишком большое по модулю "
            f"(максимум {MAX_INPUT_ABS:g}): {text}"
        )

    return complex(real, imag)


def parse_matrix_from_table(
    table_data: List[List[str]], complex_mode: bool = False
) -> np.ndarray:
    """
    Преобразует список строк таблицы в numpy-массив.

    Если complex_mode=False — возвращает вещественный массив (мнимые части
    должны быть нулевыми, иначе ValueError).
    """
    parsed = [[parse_cell(cell) for cell in row] for row in table_data]
    if complex_mode:
        return np.array(parsed, dtype=np.complex128)

    for row in parsed:
        for value in row:
            if abs(value.imag) > EPS_ZERO:
                raise ValueError(
                    "Обнаружена мнимая часть, но комплексный режим выключен"
                )
    return np.array([[v.real for v in row] for row in parsed], dtype=np.float64)


# ---------- Форматирование ----------

def format_complex(value: complex, precision: int = 4) -> str:
    """Форматирует комплексное число для вывода."""
    real = round(value.real, precision)
    imag = round(value.imag, precision)

    if abs(imag) < 10 ** (-precision):
        return f"{real:g}"
    if abs(real) < 10 ** (-precision):
        return f"{imag:g}i"
    sign = "+" if imag >= 0 else "-"
    return f"{real:g}{sign}{abs(imag):g}i"


def format_matrix(matrix: np.ndarray, precision: int = 4) -> str:
    """
    Форматирует матрицу в виде текста с ASCII-скобками и выравниванием:

        [  -0.5       0      0.5     ]
        [   0.3571  -0.2857   0.0714 ]
        [   0.2143   0.4286  -0.3571 ]
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return "[]"

    # Форматируем все ячейки
    text_cells: List[List[str]] = []
    for row in matrix:
        text_row = [format_complex(complex(v), precision) for v in row]
        text_cells.append(text_row)

    # Ширина каждого столбца — по максимальному числу в нём
    n_cols = len(text_cells[0])
    widths = [0] * n_cols
    for row in text_cells:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    # Собираем строки: сначала без правого паддинга, потом добиваем пробелами
    # до одинаковой длины — чтобы закрывающая скобка ] стояла в одной колонке.
    raw_lines: List[str] = []
    for row in text_cells:
        padded = [cell.rjust(widths[j]) for j, cell in enumerate(row)]
        inner = "  ".join(padded)   # два пробела между числами
        raw_lines.append(f"[  {inner}  ]")

    max_len = max(len(line) for line in raw_lines)
    return "\n".join(line.ljust(max_len) for line in raw_lines)


# ---------- Метод Гаусса ----------

def gauss_solve(
    A: np.ndarray, b: np.ndarray, verbose: bool = True
) -> Tuple[Optional[np.ndarray], str]:
    """Решает СЛАУ Ax = b методом Гаусса с частичным выбором ведущего элемента."""
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной")

    aug = np.hstack(
        [A.astype(np.complex128), b.reshape(-1, 1).astype(np.complex128)]
    )
    log_lines: List[str] = []

    if verbose:
        log_lines.append("Исходная расширенная матрица [A|b]:")
        log_lines.append(format_matrix(aug))
        log_lines.append("")

    swap_count = 0

    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))
        if abs(aug[pivot_row, col]) < EPS_ZERO:
            return None, "Система не имеет единственного решения (нулевой ведущий элемент)."

        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
            swap_count += 1
            if verbose:
                log_lines.append(
                    f"Шаг {col + 1}: перестановка строк {col + 1} и {pivot_row + 1}"
                )
                log_lines.append(format_matrix(aug))
                log_lines.append("")

        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            if abs(factor) < EPS_ZERO:
                continue
            aug[row] -= factor * aug[col]
            if verbose:
                log_lines.append(
                    f"Шаг {col + 1}: строка {row + 1} -= "
                    f"({format_complex(complex(factor))}) * строка {col + 1}"
                )
                log_lines.append(format_matrix(aug))
                log_lines.append("")

    for i in range(n):
        if abs(aug[i, i]) < EPS_ZERO:
            return None, "Матрица вырождена, единственного решения нет."

    if verbose:
        log_lines.append("Обратный ход:")
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        s = aug[i, n] - sum(aug[i, j] * x[j] for j in range(i + 1, n))
        x[i] = s / aug[i, i]
        if verbose:
            log_lines.append(f"x[{i + 1}] = {format_complex(complex(x[i]))}")

    log_lines.append("")
    log_lines.append(
        f"Определитель (с учётом перестановок): "
        f"{format_complex(complex(np.prod(np.diag(aug[:, :n])) * ((-1) ** swap_count)))}"
    )

    return x, "\n".join(log_lines)


# ---------- Метод Крамера ----------

def cramer_solve(A: np.ndarray, b: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Решает СЛАУ методом Крамера. Возвращает (решение, текст)."""
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Матрица должна быть квадратной")

    det_a = np.linalg.det(A)
    if abs(det_a) < EPS_ZERO:
        return None, "Определитель матрицы равен нулю — метод Крамера неприменим."

    log_lines = [f"det(A) = {format_complex(complex(det_a))}", ""]
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        Ai = A.copy().astype(np.complex128)
        Ai[:, i] = b
        det_i = np.linalg.det(Ai)
        x[i] = det_i / det_a
        log_lines.append(
            f"x[{i + 1}] = det(A{i + 1}) / det(A) = "
            f"{format_complex(complex(det_i))} / {format_complex(complex(det_a))} = "
            f"{format_complex(complex(x[i]))}"
        )
    return x, "\n".join(log_lines)


# ---------- Операции с матрицами ----------

def matrix_determinant(A: np.ndarray) -> complex:
    """Определитель квадратной матрицы. ValueError — если не квадратная."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("Определитель определён только для квадратных матриц")
    return complex(np.linalg.det(A))


def matrix_inverse(A: np.ndarray) -> np.ndarray:
    """Обратная матрица. ValueError — если не квадратная или вырождена."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("Обратная матрица существует только для квадратных матриц")
    det = np.linalg.det(A)
    if abs(det) < EPS_ZERO:
        raise ValueError("Обратной матрицы не существует (определитель равен нулю)")
    return np.linalg.inv(A)