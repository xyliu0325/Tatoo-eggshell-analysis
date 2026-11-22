"""
Sample colors from the original images and cached contours, filtering unusually dark pore pixels.
Write one Excel workbook per egg with 20 RGB measurements.
"""
import os
import re
import json
import glob
import cv2
import numpy as np
import pandas as pd


def natural_sort_key(path):
    """Sort filenames in natural numeric order."""
    basename = os.path.basename(path)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', basename)]

def filter_pores_and_get_color(img, contour):
    """
    Sample colors inside a contour after filtering unusually dark pore pixels.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

    local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted_contour = contour - np.array([x1, y1])
    cv2.drawContours(local_mask, [shifted_contour], -1, 255, -1)

    local_img = img[y1:y2, x1:x2]
    pixels = local_img[local_mask > 0]

    if len(pixels) < 10:
        return None

    brightness = 0.114 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.299 * pixels[:, 2]

    q1 = np.percentile(brightness, 25)
    q3 = np.percentile(brightness, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr

    valid_mask = brightness >= lower_bound
    valid_pixels = pixels[valid_mask]

    pore_count = len(pixels) - len(valid_pixels)
    pore_ratio = pore_count / len(pixels) * 100 if pore_count > 0 else 0

    if len(valid_pixels) < 5:
        valid_pixels = pixels

    representative_color = np.median(valid_pixels, axis=0).astype(np.uint8)
    r, g, b = int(representative_color[2]), int(representative_color[1]), int(representative_color[0])

    return (r, g, b), pore_ratio

def main():
    cache_files = glob.glob("results/*_cache.json")
    print(f"Found {len(cache_files)} cache files.")

    output_dir = "color_excel"
    os.makedirs(output_dir, exist_ok=True)

    all_eggs_data = []
    egg_number = 0

    for cache_file in sorted(cache_files, key=natural_sort_key):
        with open(cache_file, 'r', encoding='utf-8') as rf:
            cache_data = json.load(rf)

        image_path = cache_data.get('image_path', '')

        if not os.path.exists(image_path):
            base_name = os.path.basename(image_path)
            for folder in ['src/good', 'src/bad', 'src']:
                test_path = os.path.join(folder, base_name)
                if os.path.exists(test_path):
                    image_path = test_path
                    break

        if not os.path.exists(image_path):
            print(f"  Original image not found: {image_path}. Skipping.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"  Could not read image: {image_path}. Skipping.")
            continue

        egg_number += 1
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]

        result_file = cache_file.replace('_cache.json', '_result.json')
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as rf:
                result_data = json.load(rf)
            dots_data = {d['index']: d for d in result_data.get('dots', [])}
        else:
            dots_data = {d['index']: d for d in cache_data.get('dots', [])}

        print(f"Processing {image_name} (egg ID {egg_number})")

        rows = []
        for i in range(1, 21):
            if i in dots_data:
                dot = dots_data[i]
                center = dot.get('center', [None, None])
                cx, cy = center[0], center[1]

                if 'sample_contour' in dot:
                    contour = np.array(dot['sample_contour'], dtype=np.int32)
                elif 'contour' in dot:
                    contour = np.array(dot['contour'], dtype=np.int32)
                else:
                    row_data = {'Egg ID': egg_number if i == 1 else '', 'Point Number': i, 'Red': None, 'Green': None, 'Blue': None}
                    rows.append(row_data)
                    all_eggs_data.append({'Egg ID': egg_number, 'Point Number': i, 'Red': None, 'Green': None, 'Blue': None})
                    continue

                result = filter_pores_and_get_color(img, contour)
                if result:
                    (r, g, b), pore_ratio = result
                    row_data = {'Egg ID': egg_number if i == 1 else '', 'Point Number': i, 'Red': r, 'Green': g, 'Blue': b}
                    rows.append(row_data)
                    all_eggs_data.append({'Egg ID': egg_number, 'Point Number': i, 'Red': r, 'Green': g, 'Blue': b})
                    if pore_ratio > 1:
                        print(f"  Point {i}: RGB({r},{g},{b}), pore pixels excluded: {pore_ratio:.1f}%")
                else:
                    row_data = {'Egg ID': egg_number if i == 1 else '', 'Point Number': i, 'Red': None, 'Green': None, 'Blue': None}
                    rows.append(row_data)
                    all_eggs_data.append({'Egg ID': egg_number, 'Point Number': i, 'Red': None, 'Green': None, 'Blue': None})
            else:
                row_data = {'Egg ID': egg_number if i == 1 else '', 'Point Number': i, 'Red': None, 'Green': None, 'Blue': None}
                rows.append(row_data)
                all_eggs_data.append({'Egg ID': egg_number, 'Point Number': i, 'Red': None, 'Green': None, 'Blue': None})

        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f"{base_name}.xlsx")
        df.to_excel(output_path, index=False)
        print(f"  Saved: {output_path}")

    if all_eggs_data:
        summary_rows = []
        for i, row in enumerate(all_eggs_data):
            if row['Point Number'] == 1:
                summary_rows.append({'Egg ID': row['Egg ID'], 'Point Number': row['Point Number'], 'Red': row['Red'], 'Green': row['Green'], 'Blue': row['Blue']})
            else:
                summary_rows.append({'Egg ID': '', 'Point Number': row['Point Number'], 'Red': row['Red'], 'Green': row['Green'], 'Blue': row['Blue']})

        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "all_eggs_summary.xlsx")
        df_summary.to_excel(summary_path, index=False)
        print(f"\nSummary workbook saved to: {summary_path}")

    print(f"\nDone. Excel workbooks saved to {output_dir}/")

if __name__ == "__main__":
    main()
