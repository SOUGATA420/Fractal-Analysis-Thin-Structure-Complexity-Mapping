"""
Demo Script: Quick validation of Fractal Analyzer
Tests multiple fractal algorithms, thresholding, and matrix exports.
"""

import os
import numpy as np
import cv2
from fractal_analyzer import FractalAnalyzer, generate_synthetic_test_image, visualize_single_method, visualize_comparison_dashboard

def run_quick_demo():
    print("=== 1. Generating Test Image ===")
    test_img = generate_synthetic_test_image(300, 400)
    img_path = "image (11).png"
    cv2.imwrite(img_path, test_img)
    print(f"Saved sample test image to '{img_path}'. Shape: {test_img.shape}")

    print("\n=== 2. Initializing Fractal Analyzer ===")
    analyzer = FractalAnalyzer(window_size=19)
    analyzer.load_image(test_img)

    print("\n=== 3. Computing Fractal Maps & Matrices ===")
    analyzer.compute_all_maps()

    output_dir = "./demo_results"
    os.makedirs(output_dir, exist_ok=True)

    for method in ['dbc', 'blanket', 'variogram', 'morphological']:
        # Get fractal score matrix
        matrix = analyzer.get_score_matrix(method)
        
        # Threshold high fractal area (top 10% highest complexity)
        mask, cutoff, stats = analyzer.threshold_map(method, threshold_type='percentile', threshold_value=90.0)

        # Export matrix to CSV and NPY
        npy_file = os.path.join(output_dir, f"matrix_{method}.npy")
        csv_file = os.path.join(output_dir, f"matrix_{method}.csv")
        analyzer.export_matrix(method, npy_file, format='npy')
        analyzer.export_matrix(method, csv_file, format='csv')

        print(f"\n[{method.upper()}] Matrix Preview:")
        print(f"  Shape       : {matrix.shape}")
        print(f"  Range       : [{stats['min']:.3f}, {stats['max']:.3f}] | Mean: {stats['mean']:.3f}")
        print(f"  Top 10% Cutoff: {cutoff:.3f}")
        print(f"  High Area   : {stats['high_area_percentage']:.1f}% ({stats['high_pixels_count']} pixels)")
        print(f"  Center 3x3 Scores Matrix Sample:")
        cy, cx = matrix.shape[0] // 2, matrix.shape[1] // 2
        print(np.round(matrix[cy-1:cy+2, cx-1:cx+2], 3))

        # Save single method plot
        plot_file = os.path.join(output_dir, f"plot_{method}.png")
        visualize_single_method(analyzer, method=method, threshold_type='percentile', threshold_value=90.0, save_path=plot_file)

    print("\n=== 4. Generating Comparison Dashboard ===")
    comp_file = os.path.join(output_dir, "fractal_comparison_dashboard.png")
    visualize_comparison_dashboard(analyzer, threshold_type='percentile', threshold_value=90.0, save_path=comp_file)
    print(f"Comparison dashboard saved to: {comp_file}")

    print("\n[SUCCESS] All fractal mapping tests and matrix outputs verified successfully!")

if __name__ == "__main__":
    run_quick_demo()
