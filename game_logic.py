"""
Modul Game Logic & Simulasi (Core Engine) untuk bot Match-3.
Menyimulasikan deteksi match, gravitasi, dan perhitungan skor cascade.
"""

from __future__ import annotations

import copy
from typing import List, Set, Tuple

# Tipe alias untuk koordinat dan papan
Coord = Tuple[int, int]
Board = List[List[str]]

# Skor per panjang match (3, 4, 5+ dalam satu baris/kolom)
MATCH_SCORES = {3: 30, 4: 60, 5: 100}

# Karakter yang tidak bisa di-match maupun dipindahkan
BLOCKED_CHARS = frozenset({"X", "?"})


def _score_for_run(length: int) -> int:
    """Hitung poin berdasarkan panjang run match."""
    if length < 3:
        return 0
    if length in MATCH_SCORES:
        return MATCH_SCORES[length]
    # Match lebih dari 5: bonus tambahan per permen ekstra
    return MATCH_SCORES[5] + (length - 5) * 20


def _is_matchable(cell: str) -> bool:
    return cell not in BLOCKED_CHARS


def find_matches(board: Board) -> Tuple[List[Coord], int]:
    if not board or not board[0]:
        return [], 0

    rows = len(board)
    cols = len(board[0])
    matched: Set[Coord] = set()
    total_score = 0

    for r in range(rows):
        c = 0
        while c < cols:
            cell = board[r][c]
            if not _is_matchable(cell):
                c += 1
                continue

            run_start = c
            while c < cols and board[r][c] == cell:
                c += 1
            run_len = c - run_start

            if run_len >= 3:
                total_score += _score_for_run(run_len)
                for col in range(run_start, c):
                    matched.add((r, col))

    for c in range(cols):
        r = 0
        while r < rows:
            cell = board[r][c]
            if not _is_matchable(cell):
                r += 1
                continue

            run_start = r
            while r < rows and board[r][c] == cell:
                r += 1
            run_len = r - run_start

            if run_len >= 3:
                total_score += _score_for_run(run_len)
                for row in range(run_start, r):
                    matched.add((row, c))

    return sorted(matched), total_score

def apply_gravity(board: Board, matches: List[Coord]) -> Board:
    if not board:
        return board

    rows = len(board)
    cols = len(board[0])
    new_board = copy.deepcopy(board)
    match_set = set(matches)

    for r, c in match_set:
        if 0 <= r < rows and 0 <= c < cols and new_board[r][c] != "X":
            new_board[r][c] = None

    for c in range(cols):
        segments: List[Tuple[int, int]] = []
        seg_start = 0

        for r in range(rows):
            if new_board[r][c] == "X":
                if seg_start < r:
                    segments.append((seg_start, r - 1))
                seg_start = r + 1

        if seg_start < rows:
            segments.append((seg_start, rows - 1))

        for start_row, end_row in segments:
            remaining = [
                new_board[r][c]
                for r in range(start_row, end_row + 1)
                if new_board[r][c] is not None and new_board[r][c] != "X"
            ]
            segment_len = end_row - start_row + 1
            num_new = segment_len - len(remaining)

            filled = ["?"] * num_new + remaining

            for i, r in enumerate(range(start_row, end_row + 1)):
                new_board[r][c] = filled[i]

    return new_board


def _are_adjacent(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Dua sel dianggap bersebelahan jika beda 1 baris ATAU 1 kolom (bukan diagonal)."""
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)


def simulate_swap(board: Board, r1: int, c1: int, r2: int, c2: int) -> int:
    """
    Simulasikan swap dua sel bersebelahan dan hitung total skor termasuk cascade.

    Returns:
        Total skor (0 jika swap tidak valid atau tidak menghasilkan match).
    """
    rows = len(board)
    cols = len(board[0]) if board else 0

    # Validasi batas papan
    for r, c in ((r1, c1), (r2, c2)):
        if not (0 <= r < rows and 0 <= c < cols):
            return 0

    if not _are_adjacent(r1, c1, r2, c2):
        return 0

    a, b = board[r1][c1], board[r2][c2]

    # Tidak boleh menukar sel terblokir
    if a in BLOCKED_CHARS or b in BLOCKED_CHARS:
        return 0

    # Salin papan agar simulasi tidak mengubah papan asli
    sim_board = copy.deepcopy(board)
    sim_board[r1][c1], sim_board[r2][c2] = sim_board[r2][c2], sim_board[r1][c1]

    total_score = 0

    # Loop cascade: match -> gravitasi -> match lagi ...
    while True:
        matches, score = find_matches(sim_board)
        if not matches:
            break
        total_score += score
        sim_board = apply_gravity(sim_board, matches)

    return total_score
