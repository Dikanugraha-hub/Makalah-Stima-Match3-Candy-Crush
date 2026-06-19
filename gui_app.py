"""
GUI untuk mengunggah screenshot uji dan melihat usulan pergerakan Greedy
dengan panah arah (atas, bawah, kiri, kanan).
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from greedy import analyze_board
from vision import (
    VisionConfig,
    bgr_to_rgb_pil,
    detect_board,
    draw_grid_overlay,
    draw_move_arrow,
    image_to_board,
    load_image_bgr,
)

# Folder default untuk screenshot uji dan hasil analisis
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshot_uji")
HASIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_uji")

CANDY_COLORS = {
    "B": "#3B6EDC",
    "H": "#3CB85C",
    "O": "#F09028",
    "U": "#A848D8",
    "P": "#F5C0E0",
    "X": "#2AA8B0",
}


class Match3AnalyzerGUI:
    """Aplikasi GUI analisis screenshot Match-3."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Match-3 Bot — Analisis Screenshot")
        self.root.minsize(960, 640)
        self.root.configure(bg="#1e1e2e")

        self.image_bgr: Optional[np.ndarray] = None
        self.display_bgr: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.vision_config: Optional[VisionConfig] = None
        self.board = None
        self.analysis = None
        self._photo_ref: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground="#89b4fa")
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TSpinbox", font=("Segoe UI", 10))

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(main, width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        ttk.Label(sidebar, text="Match-3 Analyzer", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 12))

        ttk.Button(sidebar, text="Pilih Screenshot", command=self._pick_image).pack(fill=tk.X, pady=4)

        grid_frame = ttk.LabelFrame(sidebar, text=" Ukuran Grid (otomatis) ", padding=8)
        grid_frame.pack(fill=tk.X, pady=8)

        self.grid_size_label = ttk.Label(
            grid_frame,
            text="Belum dianalisis",
            font=("Segoe UI", 11, "bold"),
        )
        self.grid_size_label.pack(anchor=tk.W)

        ttk.Button(sidebar, text="Analisis & Cari Langkah", command=self._analyze).pack(
            fill=tk.X, pady=(12, 4)
        )

        action_frame = ttk.Frame(sidebar)
        action_frame.pack(fill=tk.X, pady=4)

        ttk.Button(action_frame, text="Reset", command=self._reset).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4)
        )
        self.download_btn = ttk.Button(
            action_frame,
            text="Unduh Hasil",
            command=self._download_result,
            state="disabled",
        )
        self.download_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self.path_label = ttk.Label(sidebar, text="Belum ada file dipilih", wraplength=230)
        self.path_label.pack(anchor=tk.W, pady=8)

        self.grid_info_label = ttk.Label(sidebar, text="ROI: —", wraplength=230)
        self.grid_info_label.pack(anchor=tk.W, pady=2)

        result_frame = ttk.LabelFrame(sidebar, text=" Usulan Pergerakan ", padding=10)
        result_frame.pack(fill=tk.X, pady=8)

        self.direction_label = tk.Label(
            result_frame,
            text="—",
            font=("Segoe UI", 28, "bold"),
            fg="#f9e2af",
            bg="#313244",
            padx=16,
            pady=12,
        )
        self.direction_label.pack(fill=tk.X)

        self.score_label = ttk.Label(result_frame, text="Skor prediksi: —")
        self.score_label.pack(anchor=tk.W, pady=(8, 0))

        self.coord_label = ttk.Label(result_frame, text="Koordinat: —")
        self.coord_label.pack(anchor=tk.W, pady=2)

        self.detail_label = ttk.Label(result_frame, text="", wraplength=230)
        self.detail_label.pack(anchor=tk.W, pady=4)

        legend = ttk.LabelFrame(sidebar, text=" Legenda ", padding=8)
        legend.pack(fill=tk.X, pady=8)
        for char, name in [
            ("B", "Biru"), ("H", "Hijau"), ("O", "Oranye"),
            ("U", "Ungu"), ("P", "Pink"), ("X", "Kosong"),
        ]:
            row = ttk.Frame(legend)
            row.pack(anchor=tk.W, pady=1)
            swatch = tk.Label(row, text="  ", bg=CANDY_COLORS[char], width=2)
            swatch.pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(row, text=f"{char} = {name}").pack(side=tk.LEFT)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        img_frame = ttk.LabelFrame(right, text=" Screenshot + Panah Arah ", padding=6)
        img_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(img_frame, bg="#11111b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        matrix_frame = ttk.LabelFrame(right, text=" Matriks Terdeteksi ", padding=6)
        matrix_frame.pack(fill=tk.X, pady=(8, 0))

        self.matrix_inner = tk.Frame(matrix_frame, bg="#313244")
        self.matrix_inner.pack()

    def _pick_image(self) -> None:
        """Dialog pemilihan file dari folder screenshot_uji."""
        initial_dir = SCREENSHOT_DIR if os.path.isdir(SCREENSHOT_DIR) else os.path.dirname(__file__)

        path = filedialog.askopenfilename(
            title="Pilih Screenshot Uji",
            initialdir=initial_dir,
            filetypes=[
                ("Gambar", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Semua file", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.image_bgr = load_image_bgr(path)
            self.display_bgr = self.image_bgr.copy()
            self.image_path = path
            self.path_label.config(text=os.path.basename(path))
            self._show_image(self.display_bgr)
            self._clear_results()
            self.grid_size_label.config(text="Belum dianalisis")
            self.grid_info_label.config(text="ROI: —")
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))

    def _show_image(self, image_bgr: np.ndarray) -> None:
        pil_img = bgr_to_rgb_pil(image_bgr)
        canvas_w = max(self.canvas.winfo_width(), 600)
        canvas_h = max(self.canvas.winfo_height(), 400)

        pil_img.thumbnail((canvas_w - 20, canvas_h - 20), Image.Resampling.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self._photo_ref,
            anchor=tk.CENTER,
        )

    def _clear_results(self) -> None:
        self.direction_label.config(text="—", fg="#f9e2af")
        self.score_label.config(text="Skor prediksi: —")
        self.coord_label.config(text="Koordinat: —")
        self.detail_label.config(text="")
        self.vision_config = None
        self.board = None
        self.analysis = None
        self.download_btn.config(state="disabled")
        for widget in self.matrix_inner.winfo_children():
            widget.destroy()

    def _reset(self) -> None:
        """Kembalikan aplikasi ke kondisi awal."""
        self.image_bgr = None
        self.display_bgr = None
        self.image_path = None
        self._photo_ref = None

        self.path_label.config(text="Belum ada file dipilih")
        self.grid_size_label.config(text="Belum dianalisis")
        self.grid_info_label.config(text="ROI: —")
        self._clear_results()

        self.canvas.delete("all")

    def _render_matrix(self, board) -> None:
        for widget in self.matrix_inner.winfo_children():
            widget.destroy()

        for r, row in enumerate(board):
            for c, char in enumerate(row):
                bg = CANDY_COLORS.get(char, "#45475a")
                fg = "#1e1e2e" if char in ("P", "H") else "#ffffff"
                lbl = tk.Label(
                    self.matrix_inner,
                    text=char,
                    width=3,
                    height=1,
                    font=("Consolas", 11, "bold"),
                    bg=bg,
                    fg=fg,
                    relief=tk.RAISED,
                    bd=1,
                )
                lbl.grid(row=r, column=c, padx=1, pady=1)

    def _analyze(self) -> None:
        if self.image_bgr is None:
            messagebox.showwarning("Peringatan", "Pilih screenshot terlebih dahulu.")
            return

        self.vision_config = detect_board(self.image_bgr)

        self.grid_size_label.config(
            text=f"{self.vision_config.rows} baris × {self.vision_config.cols} kolom"
        )

        roi = self.vision_config.roi
        if roi and roi.valid:
            self.grid_info_label.config(text=f"ROI: {roi.width}×{roi.height} px")
        else:
            self.grid_info_label.config(text="ROI: —")

        self.board = image_to_board(self.image_bgr, self.vision_config)
        self.analysis = analyze_board(self.board)

        display = draw_grid_overlay(self.image_bgr, self.vision_config, self.board)

        if self.analysis["move"]:
            r1, c1, r2, c2 = self.analysis["move"]
            display = draw_move_arrow(
                display, self.vision_config, r1, c1, r2, c2
            )

        self.display_bgr = display
        self._show_image(display)
        self._render_matrix(self.board)
        self._update_result_panel()
        self.download_btn.config(state="normal")

    def _download_result(self) -> None:
        """Simpan gambar hasil analisis ke folder hasil_uji."""
        if self.display_bgr is None or not self.image_path:
            messagebox.showwarning("Peringatan", "Belum ada hasil analisis untuk diunduh.")
            return

        input_name = os.path.basename(self.image_path)
        output_name = f"hasil_{input_name}"
        os.makedirs(HASIL_DIR, exist_ok=True)
        output_path = os.path.join(HASIL_DIR, output_name)

        if not cv2.imwrite(output_path, self.display_bgr):
            messagebox.showerror("Error", f"Gagal menyimpan gambar ke:\n{output_path}")
            return

        messagebox.showinfo(
            "Berhasil",
            f"Gambar hasil disimpan ke:\n{output_path}",
        )

    def _update_result_panel(self) -> None:
        if not self.analysis or not self.analysis["move"]:
            self.direction_label.config(text="Tidak ada\nlangkah valid", fg="#f38ba8")
            self.score_label.config(text="Skor prediksi: 0")
            self.coord_label.config(text="Koordinat: —")
            self.detail_label.config(
                text="Tidak ditemukan swap yang menghasilkan match pada papan ini."
            )
            return

        a = self.analysis
        r1, c1 = a["source"]
        r2, c2 = a["target"]
        direction = a["direction"]

        arrow_map = {
            "atas": "↑\nATAS",
            "bawah": "↓\nBAWAH",
            "kiri": "←\nKIRI",
            "kanan": "→\nKANAN",
        }
        self.direction_label.config(text=arrow_map[direction], fg="#a6e3a1")
        self.score_label.config(text=f"Skor prediksi: {a['score']}")
        self.coord_label.config(
            text=f"Sumber: ({r1}, {c1})  →  Tujuan: ({r2}, {c2})"
        )

        dir_id = {"atas": "ATAS", "bawah": "BAWAH", "kiri": "KIRI", "kanan": "KANAN"}
        candy = self.board[r1][c1]
        self.detail_label.config(
            text=(
                f"Geser permen '{candy}' di baris {r1 + 1}, kolom {c1 + 1} "
                f"ke arah {dir_id[direction]}.\n"
                f"(Indeks 0-based: ({r1},{c1}) → ({r2},{c2}))"
            )
        )


def main() -> None:
    root = tk.Tk()
    app = Match3AnalyzerGUI(root)

    def on_resize(_event):
        if app.display_bgr is not None:
            app._show_image(app.display_bgr)

    root.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
