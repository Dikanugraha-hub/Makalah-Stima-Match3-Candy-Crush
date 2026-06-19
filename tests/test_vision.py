"""Unit test deteksi otomatis ukuran grid."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision import detect_board, image_to_board, load_image_bgr

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshot_uji")


class TestGridDetection(unittest.TestCase):
    """Verifikasi deteksi otomatis ukuran matriks."""

    def test_uji_1_is_8x8(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_1.png")
        if not os.path.isfile(path):
            self.skipTest("uji_1.png tidak ditemukan")

        cfg = detect_board(load_image_bgr(path))
        self.assertEqual((cfg.rows, cfg.cols), (8, 8))

    def test_uji_4_is_6x7(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_4.png")
        if not os.path.isfile(path):
            self.skipTest("uji_4.png tidak ditemukan")

        cfg = detect_board(load_image_bgr(path))
        self.assertEqual((cfg.rows, cfg.cols), (6, 7))


class TestCellClassification(unittest.TestCase):
    """Verifikasi sel kosong (tanpa objek) diklasifikasikan sebagai X."""

    def test_uji_4_empty_center_is_all_x(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_4.png")
        if not os.path.isfile(path):
            self.skipTest("uji_4.png tidak ditemukan")

        img = load_image_bgr(path)
        cfg = detect_board(img)
        board = image_to_board(img, cfg)

        # Baris 3-5 adalah area kosong di tengah papan
        for r in range(3, 6):
            for c in range(cfg.cols):
                self.assertEqual(board[r][c], "X", f"Sel ({r},{c}) seharusnya X")

    def test_uji_4_swirl_without_candy_is_x(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_4.png")
        if not os.path.isfile(path):
            self.skipTest("uji_4.png tidak ditemukan")

        img = load_image_bgr(path)
        cfg = detect_board(img)
        board = image_to_board(img, cfg)

        # Sel swirl tanpa permen (kolom 1 baris 0-2)
        for r in range(3):
            self.assertEqual(board[r][1], "X", f"Swirl kosong ({r},1) seharusnya X")

    def test_uji_4_candy_cells_not_x(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_4.png")
        if not os.path.isfile(path):
            self.skipTest("uji_4.png tidak ditemukan")

        img = load_image_bgr(path)
        cfg = detect_board(img)
        board = image_to_board(img, cfg)

        # Sel yang jelas berisi permen
        self.assertNotEqual(board[0][3], "X")  # biru
        self.assertNotEqual(board[1][3], "X")  # ungu
        self.assertNotEqual(board[1][4], "X")  # ungu

    def test_uji_1_has_no_empty_cells(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_1.png")
        if not os.path.isfile(path):
            self.skipTest("uji_1.png tidak ditemukan")

        img = load_image_bgr(path)
        cfg = detect_board(img)
        board = image_to_board(img, cfg)

        for row in board:
            for cell in row:
                self.assertNotEqual(cell, "X")

    def test_uji_5_water_area_classified_as_x(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_5.png")
        if not os.path.isfile(path):
            self.skipTest("uji_5.png tidak ditemukan")

        img = load_image_bgr(path)
        cfg = detect_board(img)
        board = image_to_board(img, cfg)

        # Area air kosong di kiri-atas (bukan permen)
        for r in range(2):
            for c in range(5):
                self.assertEqual(board[r][c], "X", f"Air kosong ({r},{c}) seharusnya X")

        # Permen biru nyata di kolom 0 baris 4
        self.assertEqual(board[4][0], "B")

    def test_uji_6_full_matrix(self):
        path = os.path.join(SCREENSHOT_DIR, "uji_6.png")
        if not os.path.isfile(path):
            self.skipTest("uji_6.png tidak ditemukan")

        expected = [
            "XXXXXOUH",
            "XXXXXHUU",
            "XBHUOBHU",
            "XBUUOUOX",
            "BUOOXHXX",
            "XBHXXXXB",
            "XHOXXXXX",
            "XOUXXXXX",
        ]

        img = load_image_bgr(path)
        cfg = detect_board(img)
        board = image_to_board(img, cfg)

        self.assertEqual((cfg.rows, cfg.cols), (8, 8))
        for r in range(8):
            got = "".join(board[r])
            self.assertEqual(
                got,
                expected[r],
                f"Baris {r}: diharapkan {expected[r]}, didapat {got}",
            )


if __name__ == "__main__":
    unittest.main()
