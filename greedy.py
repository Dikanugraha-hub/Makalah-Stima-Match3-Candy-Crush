"""
Modul Algoritma Greedy untuk memilih langkah swap terbaik pada papan Match-3.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from game_logic import BLOCKED_CHARS, Board, simulate_swap

# Tipe kembalian: (baris_asal, kolom_asal, baris_tujuan, kolom_tujuan)
Move = Tuple[int, int, int, int]

# Label arah gerakan dalam Bahasa Indonesia
DIRECTION_LABELS = {
    "atas": "↑ ATAS",
    "bawah": "↓ BAWAH",
    "kiri": "← KIRI",
    "kanan": "→ KANAN",
}


def move_to_direction(r1: int, c1: int, r2: int, c2: int) -> str:
    """Ubah koordinat swap menjadi arah geser (atas/bawah/kiri/kanan)."""
    if r2 > r1:
        return "bawah"
    if r2 < r1:
        return "atas"
    if c2 > c1:
        return "kanan"
    return "kiri"


def analyze_board(board: Board) -> Dict:
    """
    Analisis papan: cari langkah greedy terbaik beserta skor dan arah.

    Returns:
        dict dengan kunci: move, score, direction, direction_label,
        source, target, r1, c1, r2, c2
    """
    move = find_best_move(board)
    if move is None:
        return {
            "move": None,
            "score": 0,
            "direction": None,
            "direction_label": None,
            "source": None,
            "target": None,
            "r1": None,
            "c1": None,
            "r2": None,
            "c2": None,
        }

    r1, c1, r2, c2 = move
    score = simulate_swap(board, r1, c1, r2, c2)
    direction = move_to_direction(r1, c1, r2, c2)

    return {
        "move": move,
        "score": score,
        "direction": direction,
        "direction_label": DIRECTION_LABELS[direction],
        "source": (r1, c1),
        "target": (r2, c2),
        "r1": r1,
        "c1": c1,
        "r2": r2,
        "c2": c2,
    }


def find_best_move(board: Board) -> Optional[Move]:
    """
    Iterasi seluruh papan, coba swap ke kanan dan ke bawah,
    pilih yang menghasilkan skor simulasi tertinggi.

    Returns:
        (r1, c1, r2, c2) atau None jika tidak ada langkah yang menghasilkan skor > 0.
    """
    if not board or not board[0]:
        return None

    rows = len(board)
    cols = len(board[0])

    best_score = 0
    best_move: Optional[Move] = None

    # Arah swap yang dicek: kanan dan bawah (cukup untuk mengecek semua pasangan unik)
    directions = ((0, 1), (1, 0))

    for r in range(rows):
        for c in range(cols):
            if board[r][c] in BLOCKED_CHARS:
                continue

            for dr, dc in directions:
                r2, c2 = r + dr, c + dc

                if not (0 <= r2 < rows and 0 <= c2 < cols):
                    continue
                if board[r2][c2] in BLOCKED_CHARS:
                    continue

                score = simulate_swap(board, r, c, r2, c2)

                if score > best_score:
                    best_score = score
                    best_move = (r, c, r2, c2)

    return best_move
