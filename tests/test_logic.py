"""
Unit test untuk modul game_logic dan greedy.
Menggunakan matriks dummy hardcoded termasuk area 'X'.
"""

import copy
import os
import sys
import unittest

# Agar import modul dari folder parent berjalan saat test dijalankan langsung
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game_logic import apply_gravity, find_matches, simulate_swap
from greedy import find_best_move


class TestFindMatches(unittest.TestCase):
    """Test deteksi match horizontal dan vertikal."""

    def test_horizontal_match_of_three(self):
        board = [
            ["B", "B", "B", "H"],
            ["O", "U", "H", "H"],
        ]
        coords, score = find_matches(board)
        self.assertEqual(coords, [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(score, 30)

    def test_vertical_match_of_four(self):
        board = [
            ["H", "B", "O"],
            ["H", "U", "P"],
            ["H", "B", "O"],
            ["H", "P", "U"],
        ]
        coords, score = find_matches(board)
        self.assertEqual(coords, [(0, 0), (1, 0), (2, 0), (3, 0)])
        self.assertEqual(score, 60)

    def test_ignores_x_cells(self):
        board = [
            ["B", "B", "X", "B", "B"],
            ["H", "H", "X", "O", "O"],
        ]
        # Tidak ada run >= 3 karena X memutus rangkaian
        coords, score = find_matches(board)
        self.assertEqual(coords, [])
        self.assertEqual(score, 0)

    def test_cross_match_counts_both_directions(self):
        """Match berbentuk salib: skor horizontal + vertikal."""
        board = [
            ["O", "H", "H", "H", "U"],
            ["B", "H", "P", "O", "B"],
            ["P", "H", "U", "B", "P"],
            ["U", "O", "B", "P", "O"],
        ]
        coords, score = find_matches(board)
        # Horizontal 3x H (baris 0, kolom 1-3) + vertikal 3x H (kolom 1, baris 0-2)
        # Overlap di (0,1) dan (1,1) -> total 5 sel unik
        self.assertIn((1, 1), coords)
        self.assertEqual(len(coords), 5)
        self.assertEqual(score, 60)  # 30 horizontal + 30 vertikal


class TestApplyGravity(unittest.TestCase):
    """Test gravitasi dengan obstacle 'X'."""

    def test_simple_gravity_fills_top_with_question_mark(self):
        board = [
            ["B", "H", "O"],
            ["B", "H", "U"],
            ["B", "P", "O"],
        ]
        matches = [(0, 0), (1, 0), (2, 0)]
        result = apply_gravity(board, matches)

        self.assertEqual(result[0][0], "?")
        self.assertEqual(result[1][0], "?")
        self.assertEqual(result[2][0], "?")
        # Kolom lain tidak berubah
        self.assertEqual(result[0][1], "H")

    def test_gravity_respects_x_obstacle(self):
        """Permen di atas X jatuh, tetapi tidak melewati X."""
        board = [
            ["B", "H"],
            ["O", "U"],
            ["X", "P"],
            ["H", "O"],
        ]
        # Hancurkan sel di atas X pada kolom 0
        matches = [(0, 0), (1, 0)]
        result = apply_gravity(board, matches)

        self.assertEqual(result[0][0], "?")
        self.assertEqual(result[1][0], "?")
        self.assertEqual(result[2][0], "X")  # X tetap
        self.assertEqual(result[3][0], "H")  # H di bawah X tidak naik

    def test_gravity_segment_above_x(self):
        """Gravitasi hanya dalam segmen di atas X."""
        board = [
            ["O", "B"],
            ["B", "H"],
            ["X", "O"],
        ]
        matches = [(0, 0)]
        result = apply_gravity(board, matches)

        self.assertEqual(result[0][0], "?")
        self.assertEqual(result[1][0], "B")
        self.assertEqual(result[2][0], "X")


class TestSimulateSwap(unittest.TestCase):
    """Test simulasi swap dan cascade."""

    def test_invalid_swap_with_x_returns_zero(self):
        board = [
            ["B", "X"],
            ["H", "O"],
        ]
        self.assertEqual(simulate_swap(board, 0, 0, 0, 1), 0)
        self.assertEqual(simulate_swap(board, 0, 1, 1, 1), 0)

    def test_swap_creates_horizontal_match(self):
        board = [
            ["B", "H", "B"],
            ["O", "B", "H"],
            ["U", "O", "P"],
        ]
        # Swap (0,1) H dengan (1,1) B -> baris 0: B,B,B
        score = simulate_swap(board, 0, 1, 1, 1)
        self.assertEqual(score, 30)

    def test_swap_no_match_returns_zero(self):
        board = [
            ["B", "H", "O"],
            ["U", "P", "B"],
        ]
        score = simulate_swap(board, 0, 0, 0, 1)
        self.assertEqual(score, 0)

    def test_cascade_increases_score(self):
        """Swap memicu match + gravitasi + match lagi (reaksi berantai)."""
        board = [
            ["O", "U", "H", "P"],
            ["U", "U", "O", "P"],
            ["P", "U", "P", "H"],
            ["P", "B", "U", "P"],
        ]
        # Swap (2,2) P dengan (2,3) H menghasilkan cascade > 1 putaran match
        score = simulate_swap(board, 2, 2, 2, 3)
        self.assertGreater(score, 30)

    def test_simulate_does_not_mutate_original_board(self):
        board = [
            ["B", "H", "B"],
            ["O", "B", "H"],
        ]
        original = copy.deepcopy(board)
        simulate_swap(board, 0, 1, 1, 1)
        self.assertEqual(board, original)


class TestGreedy(unittest.TestCase):
    """Test algoritma greedy find_best_move."""

    def test_finds_obvious_best_move(self):
        board = [
            ["B", "H", "B", "O"],
            ["O", "B", "H", "U"],
            ["U", "O", "P", "B"],
        ]
        move = find_best_move(board)
        self.assertIsNotNone(move)
        r1, c1, r2, c2 = move
        score = simulate_swap(board, r1, c1, r2, c2)
        self.assertGreater(score, 0)

    def test_board_with_x_finds_valid_move(self):
        board = [
            ["B", "H", "B", "X", "O"],
            ["O", "B", "H", "X", "U"],
            ["H", "O", "B", "P", "B"],
        ]
        move = find_best_move(board)
        self.assertIsNotNone(move)
        r1, c1, r2, c2 = move
        # Pastikan tidak melibatkan sel X
        self.assertNotEqual(board[r1][c1], "X")
        self.assertNotEqual(board[r2][c2], "X")

    def test_no_valid_move_returns_none(self):
        """Papan tanpa swap yang menghasilkan match."""
        board = [
            ["B", "H", "O"],
            ["U", "P", "B"],
            ["H", "O", "U"],
        ]
        move = find_best_move(board)
        self.assertIsNone(move)

    def test_picks_highest_scoring_move(self):
        """Greedy harus memilih swap dengan skor tertinggi."""
        board = [
            ["B", "B", "H", "O"],
            ["H", "O", "B", "B"],
            ["O", "H", "O", "B"],
            ["B", "O", "H", "O"],
        ]
        move = find_best_move(board)
        self.assertIsNotNone(move)

        # Hitung skor semua kemungkinan swap untuk verifikasi
        rows, cols = len(board), len(board[0])
        best_score = 0
        directions = ((0, 1), (1, 0))

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "X":
                    continue
                for dr, dc in directions:
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < rows and 0 <= c2 < cols and board[r2][c2] != "X":
                        best_score = max(best_score, simulate_swap(board, r, c, r2, c2))

        r1, c1, r2, c2 = move
        chosen_score = simulate_swap(board, r1, c1, r2, c2)
        self.assertEqual(chosen_score, best_score)


if __name__ == "__main__":
    unittest.main()
