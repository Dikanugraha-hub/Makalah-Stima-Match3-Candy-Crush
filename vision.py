"""
Modul Computer Vision: konversi screenshot menjadi matriks papan Match-3.
Mendukung deteksi otomatis ROI papan dan ukuran grid (default 8x8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Board = List[List[str]]

# Referensi warna permen dalam RGB (bukan X — X dideteksi via absence of candy)
CANDY_COLOR_REFERENCES: Dict[str, Tuple[int, int, int]] = {
    "B": (38, 155, 250),   # Biru
    "H": (62, 197, 38),    # Hijau
    "O": (251, 148, 38),   # Oranye
    "U": (196, 23, 251),   # Ungu
    "P": (245, 190, 225),  # Pink/Putih (swirl dengan objek)
}

# Referensi latar kosong (toska + swirl abu-abu)
EMPTY_BACKGROUND_RGB: Tuple[int, int, int] = (238, 235, 225)

# Ambang deteksi objek permen dalam sel (metrik diagnostik)
SAT_PIXEL_THRESHOLD = 85
DOMINANT_HUE_BIN_FRAC = 0.12

# Rentang hue OpenCV (0-180) per jenis permen / latar
HUE_ORANGE = (48, 67)
HUE_GREEN = (67, 86)
HUE_WATER = (86, 101)
HUE_BLUE = (95, 118)
HUE_PURPLE = (120, 148)
HUE_PINK_SWIRL = (135, 155)

# Ambang latar toska seragam (air kosong tanpa objek)
UNIFORM_EDGE_MAX = 0.07
UNIFORM_HUE_STD_MAX = 8.0
UNIFORM_SAT_STD_MAX = 15.0

# Ambang swirl/frosting (bukan permen)
SWIRL_LOW_SAT_FRAC = 0.92
SWIRL_MEAN_SAT_MAX = 45
FAKE_BLUE_EDGE_MAX = 0.04

# Gabungan untuk kompatibilitas konfigurasi lama
DEFAULT_COLOR_REFERENCES: Dict[str, Tuple[int, int, int]] = {
    **CANDY_COLOR_REFERENCES,
    "X": (35, 165, 175),
}


@dataclass
class BoardROI:
    """Area papan dalam piksel (x, y, lebar, tinggi)."""

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def valid(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class VisionConfig:
    """Konfigurasi pembacaan papan dari gambar."""

    rows: int = 8
    cols: int = 8
    roi: Optional[BoardROI] = None
    auto_detect: bool = True
    roi_center_ratio: float = 0.35
    color_references: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_COLOR_REFERENCES)
    )


def _classify_rgb(
    rgb: Tuple[float, float, float],
    color_refs: Dict[str, Tuple[int, int, int]],
) -> str:
    """Peta rata-rata RGB ke karakter permen terdekat."""
    best_char = "P"
    best_dist = float("inf")

    for char, ref in color_refs.items():
        dist = sum((rgb[i] - ref[i]) ** 2 for i in range(3))
        if dist < best_dist:
            best_dist = dist
            best_char = char

    return best_char


def _get_cell_patch(
    image_bgr: np.ndarray,
    roi: BoardROI,
    row: int,
    col: int,
    rows: int,
    cols: int,
    center_ratio: float = 1.0,
) -> np.ndarray:
    """Ambil area BGR sebuah sel grid (pusat atau penuh)."""
    cell_h = roi.height / rows
    cell_w = roi.width / cols

    y1 = int(roi.y + row * cell_h)
    y2 = int(roi.y + (row + 1) * cell_h)
    x1 = int(roi.x + col * cell_w)
    x2 = int(roi.x + (col + 1) * cell_w)

    if center_ratio < 1.0:
        pad_y = int((y2 - y1) * (1 - center_ratio) / 2)
        pad_x = int((x2 - x1) * (1 - center_ratio) / 2)
        y1, y2 = y1 + pad_y, y2 - pad_y
        x2, x1 = x2 - pad_x, x1 + pad_x

    return image_bgr[y1:y2, x1:x2]


def _compute_cell_metrics(patch: np.ndarray) -> Dict[str, float]:
    """Hitung metrik warna dan tekstur sebuah sel untuk klasifikasi."""
    if patch.size == 0:
        return {
            "candy_ratio": 0.0,
            "mean_sat": 0.0,
            "hue_std": 0.0,
            "sat_std": 0.0,
            "edge_ratio": 0.0,
            "dominant_hues": 0.0,
            "top_hue_frac": 0.0,
        }

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    candy_ratio = float((sat > SAT_PIXEL_THRESHOLD).mean())
    mean_sat = float(sat.mean())
    sat_std = float(sat.std())

    sat_mask = sat > 50
    if sat_mask.sum() >= 10:
        hue_std = float(hue[sat_mask].std())
        hist, _ = np.histogram(hue[sat_mask], bins=18, range=(0, 180))
        hist = hist / hist.sum()
        dominant_hues = float((hist > DOMINANT_HUE_BIN_FRAC).sum())
        top_hue_frac = float(hist.max())
    else:
        hue_std = float(hue.std())
        dominant_hues = 1.0
        top_hue_frac = 1.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edge_ratio = float(cv2.Canny(gray, 50, 150).mean() / 255.0)

    return {
        "candy_ratio": candy_ratio,
        "mean_sat": mean_sat,
        "hue_std": hue_std,
        "sat_std": sat_std,
        "edge_ratio": edge_ratio,
        "dominant_hues": dominant_hues,
        "top_hue_frac": top_hue_frac,
    }


def _is_strict_uniform_background(metrics: Dict[str, float]) -> bool:
    """Latar toska/air sangat seragam tanpa objek permen."""
    return (
        metrics["edge_ratio"] < UNIFORM_EDGE_MAX
        and metrics["hue_std"] < UNIFORM_HUE_STD_MAX
        and metrics["sat_std"] < UNIFORM_SAT_STD_MAX
    )


def _dominant_hue_stats(
    patch: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Ambil hue dominan dari piksel tersaturasi tinggi dalam sel.

    Returns:
        (mean_hue, mean_sat, low_sat_frac, mean_sat_all, edge_ratio)
        mean_hue = -1 jika tidak ada cukup piksel tersaturasi.
    """
    if patch.size == 0:
        return -1.0, 0.0, 1.0, 0.0, 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    low_frac = float((sat < 70).mean())
    mean_sat_all = float(sat.mean())

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edge = float(cv2.Canny(gray, 50, 150).mean() / 255.0)

    for thr in (100, 80, 60):
        mask = sat > thr
        if mask.sum() >= 10:
            return (
                float(hue[mask].mean()),
                float(sat[mask].mean()),
                low_frac,
                mean_sat_all,
                edge,
            )

    return -1.0, 0.0, low_frac, mean_sat_all, edge


def _is_water_background(
    metrics: Dict[str, float],
    mean_hue: float,
    edge_ratio: float,
) -> bool:
    """Deteksi sel air toska tanpa permen."""
    if _is_strict_uniform_background(metrics):
        return True
    if HUE_WATER[0] <= mean_hue <= HUE_WATER[1]:
        if edge_ratio < 0.15 or metrics["hue_std"] < 12:
            return True
    return False


def _is_swirl_blocker(
    mean_hue: float,
    mean_sat: float,
    low_sat_frac: float,
    mean_sat_all: float,
    edge_ratio: float,
) -> bool:
    """Deteksi frosting/swirl (bukan permen)."""
    if low_sat_frac > SWIRL_LOW_SAT_FRAC and mean_sat_all < SWIRL_MEAN_SAT_MAX:
        return True
    if (
        HUE_PINK_SWIRL[0] <= mean_hue <= HUE_PINK_SWIRL[1]
        and mean_sat < 95
        and low_sat_frac > 0.5
    ):
        return True
    if mean_hue > HUE_PURPLE[1] and low_sat_frac > 0.5:
        return True
    if (
        HUE_BLUE[0] <= mean_hue <= HUE_BLUE[1]
        and edge_ratio < FAKE_BLUE_EDGE_MAX
    ):
        return True
    return False


def _hue_to_candy(mean_hue: float, mean_sat: float) -> Optional[str]:
    """Peta hue dominan ke karakter permen."""
    if HUE_ORANGE[0] <= mean_hue < HUE_ORANGE[1]:
        return "O"
    if HUE_GREEN[0] <= mean_hue < HUE_GREEN[1]:
        return "H"
    if HUE_BLUE[0] <= mean_hue <= HUE_BLUE[1]:
        return "B"
    if HUE_PURPLE[0] <= mean_hue <= HUE_PURPLE[1]:
        return "U"
    if 20 <= mean_hue < HUE_ORANGE[0] and mean_sat > 55:
        return "O"
    return None


def _classify_cell_raw(patch: np.ndarray) -> str:
    """Klasifikasi satu sel berdasarkan hue HSV piksel tersaturasi tinggi."""
    if patch.size == 0:
        return "X"

    metrics = _compute_cell_metrics(patch)
    mean_hue, mean_sat, low_frac, mean_sat_all, edge = _dominant_hue_stats(patch)

    if mean_hue < 0:
        return "X"

    if _is_swirl_blocker(mean_hue, mean_sat, low_frac, mean_sat_all, edge):
        return "X"

    if _is_water_background(metrics, mean_hue, edge):
        return "X"

    candy = _hue_to_candy(mean_hue, mean_sat)
    if candy:
        return candy

    candy_pixels = patch[cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1] > 80]
    if candy_pixels.shape[0] >= 5:
        mean_bgr = candy_pixels.mean(axis=0)
        rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
        return _classify_rgb(rgb, CANDY_COLOR_REFERENCES)

    return "X"


def _content_bounds(image_bgr: np.ndarray) -> BoardROI:
    """
    Temukan bounding box area berisi permen/latar papan.
    Menggunakan piksel dengan saturasi tinggi (bukan abu-abu gelap).
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    active = (sat > 60) & (val > 40)

    rows_idx = np.where(active.any(axis=1))[0]
    cols_idx = np.where(active.any(axis=0))[0]

    if len(rows_idx) == 0 or len(cols_idx) == 0:
        return BoardROI(0, 0, w, h)

    y1, y2 = int(rows_idx[0]), int(rows_idx[-1]) + 1
    x1, x2 = int(cols_idx[0]), int(cols_idx[-1]) + 1
    return BoardROI(x1, y1, x2 - x1, y2 - y1)


def _align_roi_to_grid(roi: BoardROI, rows: int, cols: int) -> BoardROI:
    """
    Rapikan ROI agar lebar dan tinggi habis dibagi jumlah baris/kolom.
    Memangkas sedikit piksel tepi supaya setiap sel berukuran sama bulat.
    """
    x, y, bw, bh = roi.x, roi.y, roi.width, roi.height

    # Kurangi lebar/tinggi hingga kelipatan grid
    bw_aligned = bw - (bw % cols)
    bh_aligned = bh - (bh % rows)

    if bw_aligned <= 0 or bh_aligned <= 0:
        return roi

    # Pangkas berlebih secara simetris dari kanan/bawah
    trim_w = bw - bw_aligned
    trim_h = bh - bh_aligned

    return BoardROI(x, y, bw - trim_w, bh - trim_h)


def _grid_consistency_score(
    image_bgr: np.ndarray,
    roi: BoardROI,
    rows: int,
    cols: int,
) -> float:
    """
    Skor konsistensi warna antar-sel: semakin tinggi, semakin cocok ukuran grid.
    Sel permen homogen -> variansi dalam sel kecil, variansi antar sel besar.
    """
    if not roi.valid:
        return 0.0

    patch = image_bgr[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    cell_h = roi.height / rows
    cell_w = roi.width / cols
    cell_means = []
    intra_variances = []

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            cy1 = y1 + int((y2 - y1) * 0.25)
            cy2 = y2 - int((y2 - y1) * 0.25)
            cx1 = x1 + int((x2 - x1) * 0.25)
            cx2 = x2 - int((x2 - x1) * 0.25)
            roi_cell = patch[cy1:cy2, cx1:cx2]
            if roi_cell.size == 0:
                return 0.0
            cell_means.append(roi_cell.mean(axis=(0, 1)))
            intra_variances.append(roi_cell.var(axis=(0, 1)).mean())

    means = np.array(cell_means)
    between = means.std(axis=0).mean()
    within = float(np.mean(intra_variances))
    # Kombinasi: warna antar sel berbeda jelas, warna dalam sel homogen
    return float(between - 0.15 * within)


def detect_grid_size(
    image_bgr: np.ndarray,
    roi: BoardROI,
    min_size: int = 5,
    max_size: int = 10,
) -> Tuple[int, int]:
    """
    Deteksi ukuran grid (baris x kolom), mendukung papan persegi dan persegi panjang.
    Contoh: uji_1.png -> 8x8, uji_4.png -> 6x7.
    """
    best_rows, best_cols = 8, 8
    best_score = float("-inf")

    for rows in range(min_size, max_size + 1):
        for cols in range(min_size, max_size + 1):
            aligned = _align_roi_to_grid(roi, rows, cols)
            score = _grid_consistency_score(image_bgr, aligned, rows, cols)
            if score > best_score:
                best_score = score
                best_rows, best_cols = rows, cols

    return best_rows, best_cols


def detect_board(
    image_bgr: np.ndarray,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> VisionConfig:
    """
    Deteksi otomatis ROI papan dan ukuran grid dari screenshot.

    Returns:
        VisionConfig siap pakai untuk image_to_board.
    """
    content = _content_bounds(image_bgr)

    if rows is None or cols is None:
        detected_rows, detected_cols = detect_grid_size(image_bgr, content)
        rows = rows or detected_rows
        cols = cols or detected_cols
    else:
        detected_rows, detected_cols = rows, cols

    aligned = _align_roi_to_grid(content, rows, cols)

    return VisionConfig(
        rows=rows,
        cols=cols,
        roi=aligned,
        auto_detect=True,
    )


def _resolve_roi(image_bgr: np.ndarray, config: VisionConfig) -> BoardROI:
    """Tentukan ROI akhir: deteksi otomatis atau gunakan yang sudah ada."""
    h, w = image_bgr.shape[:2]

    if config.auto_detect and config.roi is None:
        detected = detect_board(image_bgr, config.rows, config.cols)
        config.rows = detected.rows
        config.cols = detected.cols
        config.roi = detected.roi

    if config.roi is None or not config.roi.valid:
        return BoardROI(0, 0, w, h)

    return config.roi


def _sample_cell_color(
    image_bgr: np.ndarray,
    roi: BoardROI,
    row: int,
    col: int,
    rows: int,
    cols: int,
    center_ratio: float,
) -> Tuple[float, float, float]:
    """Ambil rata-rata RGB dari ROI tengah sebuah sel grid."""
    cell_h = roi.height / rows
    cell_w = roi.width / cols

    cx = int(roi.x + (col + 0.5) * cell_w)
    cy = int(roi.y + (row + 0.5) * cell_h)
    half = max(2, int(min(cell_h, cell_w) * center_ratio / 2))

    y1 = max(roi.y, cy - half)
    y2 = min(roi.y + roi.height, cy + half)
    x1 = max(roi.x, cx - half)
    x2 = min(roi.x + roi.width, cx + half)

    patch = image_bgr[y1:y2, x1:x2]
    mean_bgr = patch.mean(axis=(0, 1))
    return float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])


def image_to_board(
    image_bgr: np.ndarray,
    config: Optional[VisionConfig] = None,
) -> Board:
    """
    Potong gambar menjadi grid NxM dan klasifikasikan tiap sel.

    Returns:
        Matriks 2D karakter (B, H, O, U, P, X).
    """
    if config is None:
        config = VisionConfig()

    roi = _resolve_roi(image_bgr, config)
    config.roi = roi

    board: Board = []

    for r in range(config.rows):
        row_chars: List[str] = []
        for c in range(config.cols):
            patch = _get_cell_patch(
                image_bgr,
                roi,
                r,
                c,
                config.rows,
                config.cols,
                center_ratio=0.85,
            )
            row_chars.append(_classify_cell_raw(patch))
        board.append(row_chars)

    return board


def load_image_bgr(path: str) -> np.ndarray:
    """Muat file gambar ke array BGR OpenCV."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Gagal memuat gambar: {path}")
    return image


def draw_grid_overlay(
    image_bgr: np.ndarray,
    config: VisionConfig,
    board: Optional[Board] = None,
) -> np.ndarray:
    """Gambar garis grid dan label karakter di atas area papan."""
    overlay = image_bgr.copy()
    roi = _resolve_roi(image_bgr, config)
    config.roi = roi

    rows, cols = config.rows, config.cols
    cell_h = roi.height / rows
    cell_w = roi.width / cols

    for r in range(rows + 1):
        y = int(roi.y + r * cell_h)
        cv2.line(
            overlay,
            (roi.x, y),
            (roi.x + roi.width, y),
            (255, 255, 255),
            1,
        )

    for c in range(cols + 1):
        x = int(roi.x + c * cell_w)
        cv2.line(
            overlay,
            (x, roi.y),
            (x, roi.y + roi.height),
            (255, 255, 255),
            1,
        )

    if board:
        for r in range(rows):
            for c in range(cols):
                cx = int(roi.x + (c + 0.5) * cell_w)
                cy = int(roi.y + (r + 0.5) * cell_h)
                label = board[r][c]
                cv2.putText(
                    overlay,
                    label,
                    (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    label,
                    (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return overlay


def draw_move_arrow(
    image_bgr: np.ndarray,
    config: VisionConfig,
    r1: int,
    c1: int,
    r2: int,
    c2: int,
) -> np.ndarray:
    """Gambar panah berarah dari sel sumber ke sel tujuan swap."""
    result = image_bgr.copy()
    roi = _resolve_roi(image_bgr, config)
    config.roi = roi

    rows, cols = config.rows, config.cols
    cell_h = roi.height / rows
    cell_w = roi.width / cols

    x1 = int(roi.x + (c1 + 0.5) * cell_w)
    y1 = int(roi.y + (r1 + 0.5) * cell_h)
    x2 = int(roi.x + (c2 + 0.5) * cell_w)
    y2 = int(roi.y + (r2 + 0.5) * cell_h)

    radius = int(min(cell_h, cell_w) * 0.32)
    cv2.circle(result, (x1, y1), radius, (0, 255, 255), 3)
    cv2.arrowedLine(
        result,
        (x1, y1),
        (x2, y2),
        (0, 220, 255),
        thickness=4,
        tipLength=0.35,
    )

    return result


def bgr_to_rgb_pil(image_bgr: np.ndarray):
    """Konversi BGR numpy ke Image PIL RGB."""
    from PIL import Image

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
