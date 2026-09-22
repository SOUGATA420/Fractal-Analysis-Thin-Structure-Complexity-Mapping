"""
Fractal Analysis & Mapping Engine
=================================
Performs multi-method local fractal dimension mapping on 2D images,
detects and isolates high fractal score regions via customizable thresholding,
and generates/exports numeric fractal score matrices.

Supported Methods:
1. Differential Box Counting (DBC) - Sarkar & Chaudhuri (1994)
2. Blanket Method - Peleg et al. (1984)
3. Variogram / Hurst Exponent Map (Fractional Brownian Motion)
4. Morphological / Boundary Box-Counting Map


"""

import os
import argparse
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


# ==============================================================================
# 1. CORE FRACTAL DIMENSION ALGORITHMS (VECTORIZED & OPTIMIZED)
# ==============================================================================

def compute_dbc_map(
    gray: np.ndarray,
    window_size: int = 21,
    scales: Optional[List[int]] = None
) -> np.ndarray:
    """
    Differential Box Counting (DBC) Fractal Dimension Map.
    Treats grayscale intensity as a 3D topographic surface (Sarkar & Chaudhuri).
    Returns a 2D matrix of fractal dimensions in range [2.0, 3.0].
    """
    if scales is None:
        scales = [3, 5, 7, 9, 13]

    h, w = gray.shape
    gray_f = gray.astype(np.float64)
    log_inv_s = []
    log_n_s = []

    # Maximum gray range
    g_max = 256.0

    for s in scales:
        # Structuring element size s x s
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
        local_max = cv2.dilate(gray_f, kernel).astype(np.float64)
        local_min = cv2.erode(gray_f, kernel).astype(np.float64)

        # Height of box for scale s: s' = s * (G / window_size)
        box_h = max(1.0, s * (g_max / float(window_size)))
        
        # Differential box count per column
        n_column = np.ceil(local_max / box_h) - np.ceil(local_min / box_h) + 1.0

        # Sum of boxes over the local sliding window:
        # At scale s, the window contains (window_size / s)^2 grid cells.
        # uniform_filter calculates the mean column count across the window.
        # Multiplying by (window_size / s)^2 gives the total box count covering the 3D surface.
        n_window = ((window_size / float(s)) ** 2) * uniform_filter(n_column, size=window_size, mode='reflect')
        n_window = np.maximum(n_window, 1.0)

        log_inv_s.append(np.log(1.0 / s))
        log_n_s.append(np.log(n_window))

    log_inv_s = np.array(log_inv_s) # (K,)
    log_n_s = np.stack(log_n_s, axis=0) # (K, H, W)

    # Vectorized least squares linear regression slope: slope = Cov(X, Y) / Var(X)
    x_mean = np.mean(log_inv_s)
    x_diff = log_inv_s - x_mean
    var_x = np.sum(x_diff ** 2)

    y_mean = np.mean(log_n_s, axis=0) # (H, W)
    y_diff = log_n_s - y_mean[None, :, :] # (K, H, W)

    cov_xy = np.sum(x_diff[:, None, None] * y_diff, axis=0)
    dbc_map = cov_xy / (var_x + 1e-12)

    # Clamp to theoretical range for 2D surface: [2.0, 3.0]
    dbc_map = np.clip(dbc_map, 2.0, 3.0)
    return dbc_map


def compute_blanket_map(
    gray: np.ndarray,
    window_size: int = 21,
    max_epsilon: int = 5
) -> np.ndarray:
    """
    Peleg et al. Blanket Method Fractal Dimension Map.
    Simulates morphological dilation/erosion blankets over thickness epsilon.
    Area A(eps) = (V(eps) - V(eps-1)) / 2.
    Fitting log(A(eps)) vs log(eps) gives slope = 2 - FD => FD = 2 - slope.
    """
    gray_f = gray.astype(np.float64)
    u_prev = gray_f.copy()
    b_prev = gray_f.copy()

    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    epsilons = list(range(1, max_epsilon + 1))
    log_eps = []
    log_area = []

    v_prev = np.zeros_like(gray_f)

    for eps in epsilons:
        # Upper surface dilation + 1
        u_curr = np.maximum(u_prev + 1.0, cv2.dilate(u_prev, kernel_cross))
        # Lower surface erosion - 1
        b_curr = np.minimum(b_prev - 1.0, cv2.erode(b_prev, kernel_cross))

        # Local volume in window
        thickness = u_curr - b_curr
        v_curr = uniform_filter(thickness, size=window_size, mode='reflect') * (window_size ** 2)

        # Surface area
        area = (v_curr - v_prev) / 2.0
        area = np.maximum(area, 1e-3)

        log_eps.append(np.log(float(eps)))
        log_area.append(np.log(area))

        u_prev = u_curr
        b_prev = b_curr
        v_prev = v_curr

    log_eps = np.array(log_eps)
    log_area = np.stack(log_area, axis=0) # (K, H, W)

    # Vectorized least squares slope: log(A) ~ (2 - FD) * log(eps) + C
    x_mean = np.mean(log_eps)
    x_diff = log_eps - x_mean
    var_x = np.sum(x_diff ** 2)

    y_mean = np.mean(log_area, axis=0)
    y_diff = log_area - y_mean[None, :, :]

    cov_xy = np.sum(x_diff[:, None, None] * y_diff, axis=0)
    slope = cov_xy / (var_x + 1e-12)

    blanket_map = 2.0 - slope
    blanket_map = np.clip(blanket_map, 2.0, 3.0)
    return blanket_map


def compute_variogram_map(
    gray: np.ndarray,
    window_size: int = 21,
    lags: Optional[List[int]] = None
) -> np.ndarray:
    """
    Variogram / Hurst Exponent Fractal Map based on Fractional Brownian Motion (fBm).
    Structure function: gamma(lag) = E[(I(x+lag) - I(x))^2] ~ lag^(2H)
    Fractal Dimension: FD = 3 - H.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 6]

    gray_f = gray.astype(np.float64)
    log_lags = []
    log_gamma = []

    for lag in lags:
        # Difference in horizontal and vertical directions
        diff_h = np.zeros_like(gray_f)
        diff_v = np.zeros_like(gray_f)

        diff_h[:, :-lag] = (gray_f[:, lag:] - gray_f[:, :-lag]) ** 2
        diff_v[:-lag, :] = (gray_f[lag:, :] - gray_f[:-lag, :]) ** 2

        sq_diff = 0.5 * (diff_h + diff_v)
        local_gamma = uniform_filter(sq_diff, size=window_size, mode='reflect')
        local_gamma = np.maximum(local_gamma, 1e-4)

        log_lags.append(np.log(float(lag)))
        log_gamma.append(np.log(local_gamma))

    log_lags = np.array(log_lags)
    log_gamma = np.stack(log_gamma, axis=0)

    x_mean = np.mean(log_lags)
    x_diff = log_lags - x_mean
    var_x = np.sum(x_diff ** 2)

    y_mean = np.mean(log_gamma, axis=0)
    y_diff = log_gamma - y_mean[None, :, :]

    cov_xy = np.sum(x_diff[:, None, None] * y_diff, axis=0)
    # Slope = 2 * Hurst
    slope = cov_xy / (var_x + 1e-12)
    hurst = np.clip(slope / 2.0, 0.01, 0.99)
    fd_map = 3.0 - hurst
    return np.clip(fd_map, 2.0, 3.0)


def compute_morphological_box_map(
    gray: np.ndarray,
    window_size: int = 21,
    scales: Optional[List[int]] = None
) -> np.ndarray:
    """
    Morphological Edge & Structural Box-Counting Dimension Map.
    Evaluates boundary and texture complexity: FD in [1.0, 2.0].
    """
    if scales is None:
        scales = [2, 4, 6, 8, 12]

    # Compute high-frequency morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel).astype(np.float64)
    
    # Adaptive local thresholding to create binary edge/texture map
    norm_grad = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(norm_grad, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary.astype(np.float64)

    log_inv_s = []
    log_counts = []

    for s in scales:
        s_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
        # Dilate binary points with box size s
        covered = cv2.dilate(binary, s_kernel)
        # Fraction of covered space in window multiplied by (window_size / s)^2
        n_boxes = ((window_size / float(s)) ** 2) * uniform_filter(covered, size=window_size, mode='reflect')
        n_boxes = np.maximum(n_boxes, 1.0)

        log_inv_s.append(np.log(1.0 / s))
        log_counts.append(np.log(n_boxes))

    log_inv_s = np.array(log_inv_s)
    log_counts = np.stack(log_counts, axis=0)

    x_mean = np.mean(log_inv_s)
    x_diff = log_inv_s - x_mean
    var_x = np.sum(x_diff ** 2)

    y_mean = np.mean(log_counts, axis=0)
    y_diff = log_counts - y_mean[None, :, :]

    cov_xy = np.sum(x_diff[:, None, None] * y_diff, axis=0)
    fd_map = cov_xy / (var_x + 1e-12)
    return np.clip(fd_map, 1.0, 2.0)


# ==============================================================================
# 2. THRESHOLDING & HIGH FRACTAL AREA EXTRACTION
# ==============================================================================

def threshold_fractal_map(
    fractal_matrix: np.ndarray,
    threshold_type: str = 'percentile',
    threshold_value: float = 85.0
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Thresholds the fractal score matrix to isolate high-fractal regions.

    Parameters:
    - fractal_matrix: 2D numpy array of fractal scores.
    - threshold_type: 'percentile' (e.g. 85 = top 15%), 'absolute' (e.g. 2.65), or 'otsu'.
    - threshold_value: cutoff parameter.

    Returns:
    - mask: boolean 2D numpy array (True for high-fractal pixels).
    - cutoff_value: numerical threshold applied.
    - stats: dictionary of summary statistics.
    """
    flat = fractal_matrix.flatten()
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    mean_val = float(np.mean(flat))
    median_val = float(np.median(flat))
    std_val = float(np.std(flat))

    if threshold_type == 'percentile':
        cutoff_value = float(np.percentile(flat, threshold_value))
    elif threshold_type == 'absolute':
        cutoff_value = float(threshold_value)
    elif threshold_type == 'otsu':
        # Normalize to uint8 for Otsu
        norm_map = cv2.normalize(fractal_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        otsu_th, _ = cv2.threshold(norm_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cutoff_value = min_val + (otsu_th / 255.0) * (max_val - min_val)
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}. Use 'percentile', 'absolute', or 'otsu'.")

    mask = fractal_matrix >= cutoff_value
    high_count = int(np.sum(mask))
    total_pixels = int(fractal_matrix.size)
    high_pct = (high_count / total_pixels) * 100.0

    stats = {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'threshold_cutoff': cutoff_value,
        'high_pixels_count': high_count,
        'total_pixels': total_pixels,
        'high_area_percentage': high_pct,
    }
    return mask, cutoff_value, stats


def create_highlight_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    fractal_matrix: np.ndarray,
    colormap_name: str = 'inferno',
    alpha: float = 0.65,
    dim_background: float = 0.35
) -> np.ndarray:
    """
    Creates a visual overlay highlighting only the high-fractal score area.
    Unselected background is dimmed, while high-fractal regions glow with a colormap
    and crisp boundary contours.
    """
    if len(image.shape) == 2:
        base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image.copy()

    # Dim the background image
    dimmed_rgb = (base_rgb * dim_background).astype(np.uint8)

    # Normalize fractal matrix to [0, 1]
    f_min, f_max = fractal_matrix.min(), fractal_matrix.max()
    if f_max > f_min:
        norm_map = (fractal_matrix - f_min) / (f_max - f_min)
    else:
        norm_map = np.zeros_like(fractal_matrix)

    # Get colored heatmap
    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = (cmap(norm_map)[:, :, :3] * 255).astype(np.uint8)

    # Blend colored heatmap only over masked high-fractal area
    output = dimmed_rgb.copy()
    high_area_blend = (
        alpha * colored_heatmap[mask] + (1.0 - alpha) * base_rgb[mask]
    ).astype(np.uint8)
    output[mask] = high_area_blend

    # Draw contours of high fractal regions
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 230), 1)  # Cyan glowing border

    return output


# ==============================================================================
# 3. FRACTAL ANALYZER ORCHESTRATOR CLASS
# ==============================================================================

class FractalAnalyzer:
    """
    Comprehensive pipeline to compute fractal maps, isolate high fractal zones,
    and inspect/export the resulting fractal score matrices.
    """
    SUPPORTED_METHODS = {
        'dbc': compute_dbc_map,
        'blanket': compute_blanket_map,
        'variogram': compute_variogram_map,
        'morphological': compute_morphological_box_map,
    }

    METHOD_TITLES = {
        'dbc': "Differential Box Counting (DBC)",
        'blanket': "Blanket Surface Method",
        'variogram': "Variogram / Hurst Fractal Map",
        'morphological': "Morphological Box-Counting Map",
    }

    def __init__(self, window_size: int = 21):
        self.window_size = window_size
        self.raw_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.fractal_matrices: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}
        self.stats: Dict[str, Dict[str, float]] = {}

    def load_image(self, image_path_or_array: Union[str, np.ndarray]) -> np.ndarray:
        """Loads an image from filepath or accepts a numpy array."""
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Image not found at '{image_path_or_array}'")
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise ValueError(f"Could not decode image at '{image_path_or_array}'")
            self.raw_image = img
            self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_path_or_array, np.ndarray):
            self.raw_image = image_path_or_array
            if len(image_path_or_array.shape) == 2:
                self.gray_image = image_path_or_array.copy()
            else:
                self.gray_image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Input must be a filepath string or numpy array.")
        return self.gray_image

    def compute_map(self, method: str = 'dbc') -> np.ndarray:
        """Computes a specific fractal map."""
        if self.gray_image is None:
            raise RuntimeError("No image loaded. Call load_image() first.")
        method = method.lower()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method '{method}'. Supported: {list(self.SUPPORTED_METHODS.keys())}")
        
        matrix = self.SUPPORTED_METHODS[method](self.gray_image, window_size=self.window_size)
        self.fractal_matrices[method] = matrix
        return matrix

    def compute_all_maps(self) -> Dict[str, np.ndarray]:
        """Computes all available fractal mapping algorithms."""
        for m in self.SUPPORTED_METHODS:
            self.compute_map(m)
        return self.fractal_matrices

    def threshold_map(
        self,
        method: str = 'dbc',
        threshold_type: str = 'percentile',
        threshold_value: float = 85.0
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """Applies thresholding to isolate high fractal score zones."""
        if method not in self.fractal_matrices:
            self.compute_map(method)
        matrix = self.fractal_matrices[method]
        mask, cutoff, stats = threshold_fractal_map(matrix, threshold_type, threshold_value)
        self.masks[method] = mask
        self.stats[method] = stats
        return mask, cutoff, stats

    def get_score_matrix(self, method: str = 'dbc') -> np.ndarray:
        """Returns the 2D fractal score matrix for a given method."""
        if method not in self.fractal_matrices:
            self.compute_map(method)
        return self.fractal_matrices[method]

    def export_matrix(self, method: str, output_path: str, format: str = 'npy') -> str:
        """Exports the fractal score matrix to .npy, .csv, or .txt."""
        matrix = self.get_score_matrix(method)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if format == 'npy':
            np.save(output_path, matrix)
        elif format in ['csv', 'txt']:
            delimiter = ',' if format == 'csv' else '\t'
            np.savetxt(output_path, matrix, delimiter=delimiter, fmt='%.4f')
        else:
            raise ValueError("format must be 'npy', 'csv', or 'txt'")
        return output_path


# ==============================================================================
# 4. VISUALIZATION & REPORTING
# ==============================================================================

def visualize_single_method(
    analyzer: FractalAnalyzer,
    method: str = 'dbc',
    threshold_type: str = 'percentile',
    threshold_value: float = 85.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates an informative 4-panel visual dashboard for a single fractal method:
    1. Original Image
    2. Fractal Score Matrix (Heatmap with Colorbar)
    3. High Fractal Area Binary Mask
    4. Highlighted Regions Overlay with Contours
    """
    matrix = analyzer.get_score_matrix(method)
    mask, cutoff, stats = analyzer.threshold_map(method, threshold_type, threshold_value)
    overlay = create_highlight_overlay(analyzer.raw_image, mask, matrix)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor='#0d1117')
    plt.subplots_adjust(wspace=0.18, hspace=0.25)

    title_color = '#e6edf3'
    sub_color = '#58a6ff'

    # Panel 1: Original Image
    ax0 = axes[0, 0]
    ax0.set_facecolor('#161b22')
    ax0.imshow(cv2.cvtColor(analyzer.raw_image, cv2.COLOR_BGR2RGB) if len(analyzer.raw_image.shape) == 3 else analyzer.raw_image, cmap='gray')
    ax0.set_title("Input Image", color=title_color, fontsize=13, pad=8, weight='bold')
    ax0.axis('off')

    # Panel 2: Fractal Score Matrix Heatmap
    ax1 = axes[0, 1]
    ax1.set_facecolor('#161b22')
    im1 = ax1.imshow(matrix, cmap='magma')
    ax1.set_title(f"{analyzer.METHOD_TITLES.get(method, method)} Matrix", color=title_color, fontsize=13, pad=8, weight='bold')
    ax1.axis('off')
    cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=title_color)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=title_color)
    cbar.set_label('Fractal Dimension (FD)', color=title_color, fontsize=10)

    # Panel 3: High Fractal Region Mask
    ax2 = axes[1, 0]
    ax2.set_facecolor('#161b22')
    ax2.imshow(mask, cmap='Blues_r', vmin=0, vmax=1)
    ax2.set_title(f"High-Fractal Mask (Cutoff: {cutoff:.3f}, {stats['high_area_percentage']:.1f}% area)", 
                  color=title_color, fontsize=13, pad=8, weight='bold')
    ax2.axis('off')

    # Panel 4: Composite Highlight Overlay
    ax3 = axes[1, 1]
    ax3.set_facecolor('#161b22')
    ax3.imshow(overlay)
    ax3.set_title(f"High-Fractal Area Highlight (Threshold: {threshold_type} {threshold_value})", 
                  color=sub_color, fontsize=13, pad=8, weight='bold')
    ax3.axis('off')

    fig.suptitle(f"Fractal Analysis — {analyzer.METHOD_TITLES.get(method, method)}", 
                 color='#ffffff', fontsize=16, weight='bold', y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


def visualize_comparison_dashboard(
    analyzer: FractalAnalyzer,
    threshold_type: str = 'percentile',
    threshold_value: float = 85.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates a multi-method comparison grid showcasing all fractal mapping techniques side-by-side.
    """
    methods = list(analyzer.SUPPORTED_METHODS.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, 4, figsize=(18, 4.2 * n_methods), facecolor='#0d1117')
    plt.subplots_adjust(wspace=0.15, hspace=0.30)
    title_color = '#e6edf3'

    for row_idx, method in enumerate(methods):
        matrix = analyzer.get_score_matrix(method)
        mask, cutoff, stats = analyzer.threshold_map(method, threshold_type, threshold_value)
        overlay = create_highlight_overlay(analyzer.raw_image, mask, matrix)

        # 1. Original
        axes[row_idx, 0].imshow(cv2.cvtColor(analyzer.raw_image, cv2.COLOR_BGR2RGB) if len(analyzer.raw_image.shape) == 3 else analyzer.raw_image, cmap='gray')
        axes[row_idx, 0].set_title(f"Original ({analyzer.METHOD_TITLES[method]})", color=title_color, fontsize=11, pad=6)
        axes[row_idx, 0].axis('off')

        # 2. Fractal Score Matrix Heatmap
        im_heat = axes[row_idx, 1].imshow(matrix, cmap='turbo')
        axes[row_idx, 1].set_title(f"Score Matrix [Min:{stats['min']:.2f}, Max:{stats['max']:.2f}]", color=title_color, fontsize=11, pad=6)
        axes[row_idx, 1].axis('off')
        cb = fig.colorbar(im_heat, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=title_color)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=title_color)

        # 3. Mask
        axes[row_idx, 2].imshow(mask, cmap='copper')
        axes[row_idx, 2].set_title(f"High-Score Mask (Cutoff: {cutoff:.3f})", color=title_color, fontsize=11, pad=6)
        axes[row_idx, 2].axis('off')

        # 4. Highlight Overlay
        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title(f"High Fractal Isolation ({stats['high_area_percentage']:.1f}% area)", color='#58a6ff', fontsize=11, pad=6)
        axes[row_idx, 3].axis('off')

    fig.suptitle(f"Multi-Method Fractal Analysis & High-Complexity Region Detection", 
                 color='#ffffff', fontsize=17, weight='bold', y=0.995)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    return fig


# ==============================================================================
# 5. SYNTHETIC DEMONSTRATION IMAGE GENERATOR
# ==============================================================================

def generate_synthetic_test_image(height: int = 360, width: int = 480) -> np.ndarray:
    """
    Generates a rich benchmark image featuring:
    - Smooth gradient region (low fractal dimension, FD ~ 2.0)
    - Perlin-like noisy texture (medium fractal dimension, FD ~ 2.3)
    - Mandelbrot boundary structure (high fractal dimension, FD ~ 2.8)
    - Sharp geometric edges
    """
    canvas = np.zeros((height, width), dtype=np.float32)

    # 1. Background smooth gradient (left to right)
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    smooth_gradient = 80.0 * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    canvas += smooth_gradient

    # 2. Add textured noise patch in the top-right
    tr_y1, tr_y2 = 20, height // 2 - 10
    tr_x1, tr_x2 = width // 2 + 20, width - 20
    noise_patch = np.random.normal(128, 40, (tr_y2 - tr_y1, tr_x2 - tr_x1))
    noise_patch = cv2.GaussianBlur(noise_patch, (5, 5), 1.2)
    canvas[tr_y1:tr_y2, tr_x1:tr_x2] = noise_patch

    # 3. Add Mandelbrot fractal patch in the center-left
    m_h, m_w = height - 60, width // 2 - 30
    cx = np.linspace(-2.0, 0.6, m_w)
    cy = np.linspace(-1.2, 1.2, m_h)
    c_real, c_imag = np.meshgrid(cx, cy)
    c = c_real + 1j * c_imag
    z = np.zeros_like(c)
    fractal_iter = np.zeros((m_h, m_w), dtype=np.float32)
    max_iter = 60

    for i in range(max_iter):
        diverge = np.abs(z) > 2.0
        fractal_iter[~diverge] = i
        z[~diverge] = z[~diverge] ** 2 + c[~diverge]

    # Normalize Mandelbrot into [20, 240]
    fractal_iter = cv2.normalize(fractal_iter, None, 20, 240, cv2.NORM_MINMAX)
    canvas[30:30 + m_h, 20:20 + m_w] = fractal_iter

    # 4. Add geometric rings in bottom-right
    cr_y, cr_x = int(height * 0.75), int(width * 0.75)
    for r in range(10, 60, 8):
        cv2.circle(canvas, (cr_x, cr_y), r, 220, 2)

    # Normalize to uint8 image
    canvas_u8 = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return canvas_u8


# ==============================================================================
# 6. CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute image fractal maps, isolate high fractal score areas, and export fractal score matrices."
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image file.")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode using a synthetic fractal benchmark image.")
    parser.add_argument("--method", type=str, default="all", choices=["dbc", "blanket", "variogram", "morphological", "all"],
                        help="Fractal mapping method to compute (default: all).")
    parser.add_argument("--window-size", type=int, default=21, help="Sliding window size for local fractal estimation (default: 21).")
    parser.add_argument("--threshold-type", type=str, default="percentile", choices=["percentile", "absolute", "otsu"],
                        help="Thresholding strategy: 'percentile', 'absolute', or 'otsu' (default: percentile).")
    parser.add_argument("--threshold-val", type=float, default=85.0,
                        help="Threshold value (e.g., 85.0 for top 15% percentile, or absolute FD score like 2.6).")
    parser.add_argument("--output-dir", type=str, default="./fractal_output", help="Directory to save matrices and plots.")
    parser.add_argument("--export-format", type=str, default="npy", choices=["npy", "csv"],
                        help="Matrix export format: 'npy' or 'csv' (default: npy).")
    parser.add_argument("--save-output", action="store_true", default=True, help="Save visualization figures and matrices.")

    args = parser.parse_args()

    # Determine input image
    if args.demo or args.image is None:
        print("[*] Generating synthetic fractal test benchmark image...")
        input_image = generate_synthetic_test_image()
        image_name = "synthetic_demo"
    else:
        print(f"[*] Loading input image from: {args.image}")
        input_image = args.image
        image_name = os.path.splitext(os.path.basename(args.image))[0]

    analyzer = FractalAnalyzer(window_size=args.window_size)
    analyzer.load_image(input_image)

    os.makedirs(args.output_dir, exist_ok=True)

    methods_to_run = list(analyzer.SUPPORTED_METHODS.keys()) if args.method == "all" else [args.method]

    print(f"[*] Computing fractal maps using window size = {args.window_size}...")
    for m in methods_to_run:
        matrix = analyzer.compute_map(m)
        mask, cutoff, stats = analyzer.threshold_map(m, args.threshold_type, args.threshold_val)

        print(f"\n========================================================")
        print(f" Method: {analyzer.METHOD_TITLES.get(m, m).upper()}")
        print(f"========================================================")
        print(f" - Fractal Score Matrix Shape: {matrix.shape}")
        print(f" - Min Fractal Score        : {stats['min']:.4f}")
        print(f" - Max Fractal Score        : {stats['max']:.4f}")
        print(f" - Mean Fractal Score       : {stats['mean']:.4f}")
        print(f" - Median Fractal Score     : {stats['median']:.4f}")
        print(f" - Std Deviation            : {stats['std']:.4f}")
        print(f" - Applied Threshold Cutoff : {cutoff:.4f} ({args.threshold_type}: {args.threshold_val})")
        print(f" - High Fractal Area Ratio  : {stats['high_area_percentage']:.2f}% ({stats['high_pixels_count']:,} pixels)")

        # Export Matrix
        if args.save_output:
            matrix_path = os.path.join(args.output_dir, f"{image_name}_{m}_matrix.{args.export_format}")
            analyzer.export_matrix(m, matrix_path, format=args.export_format)
            print(f" [+] Exported score matrix to: {matrix_path}")

            # Single method plot
            plot_path = os.path.join(args.output_dir, f"{image_name}_{m}_analysis.png")
            visualize_single_method(analyzer, method=m, threshold_type=args.threshold_type, 
                                    threshold_value=args.threshold_val, save_path=plot_path)
            print(f" [+] Saved visual analysis to: {plot_path}")

    # Full comparison dashboard if all methods requested
    if args.method == "all" and args.save_output:
        comparison_path = os.path.join(args.output_dir, f"{image_name}_comparison_dashboard.png")
        visualize_comparison_dashboard(analyzer, threshold_type=args.threshold_type, 
                                       threshold_value=args.threshold_val, save_path=comparison_path)
        print(f"\n[+] Saved multi-method comparison dashboard to: {comparison_path}")

    print("\n[SUCCESS] Fractal analysis completed successfully!")


if __name__ == "__main__":
    main()
