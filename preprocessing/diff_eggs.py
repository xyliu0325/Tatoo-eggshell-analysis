"""
Compare RGB measurements for matching eggs in two Excel workbooks.
Read 0.xlsx and 3.xlsx from the script directory.
Write absolute and enhanced RGB differences and color maps for each egg.
"""
import os
import cv2
import numpy as np
import pandas as pd

def draw_color_map(colors, save_path):
    """Draw a color-difference map on a black 4 x 5 grid without labels."""
    H, W = 2000, 2500
    R = 180
    n_rows, n_cols = 4, 5

    row_pos = [int(H / (n_rows + 1) * (i + 1)) for i in range(n_rows)]
    col_pos = [int(W / (n_cols + 1) * (i + 1)) for i in range(n_cols)]

    img = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(min(20, len(colors))):
        row = i // n_cols
        col = i % n_cols
        center = (col_pos[col], row_pos[row])

        if colors[i] is not None:
            b, g, r = colors[i]
            cv2.circle(img, center, R, (b, g, r), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, center, R, (50, 50, 50), 2, cv2.LINE_AA)

    _, buf = cv2.imencode('.png', img)
    with open(save_path, 'wb') as f:
        f.write(buf.tobytes())

def compute_diff(row1, row2, method='abs'):
    """Calculate the RGB difference for one point."""
    r1, g1, b1 = row1['Red'], row1['Green'], row1['Blue']
    r2, g2, b2 = row2['Red'], row2['Green'], row2['Blue']

    if pd.isna(r1) or pd.isna(r2):
        return None, None, None

    if method == 'abs':
        dr = abs(int(r1) - int(r2))
        dg = abs(int(g1) - int(g2))
        db = abs(int(b1) - int(b2))
    else:

        dr = min(255, (abs(int(r1) - int(r2)) * 255) // 130)
        dg = min(255, (abs(int(g1) - int(g2)) * 255) // 130)
        db = min(255, (abs(int(b1) - int(b2)) * 255) // 130)
    return dr, dg, db

def main():
    input_dir = os.path.dirname(os.path.abspath(__file__))
    abs_dir = os.path.join(input_dir, "abs")
    enh_dir = os.path.join(input_dir, "enhanced")
    os.makedirs(abs_dir, exist_ok=True)
    os.makedirs(enh_dir, exist_ok=True)

    df1 = pd.read_excel(os.path.join(input_dir, "0.xlsx"))
    df2 = pd.read_excel(os.path.join(input_dir, "3.xlsx"))

    legacy_headers = {'\u86cb\u53f7': 'Egg ID', '\u5e8f\u53f7': 'Point Number'}
    df1 = df1.rename(columns=legacy_headers)
    df2 = df2.rename(columns=legacy_headers)

    df1['Egg ID'] = df1['Egg ID'].ffill()
    df2['Egg ID'] = df2['Egg ID'].ffill()

    eggs1 = df1['Egg ID'].unique()
    eggs2 = df2['Egg ID'].unique()
    common_eggs = sorted(set(eggs1) & set(eggs2))

    print(f"Workbook 1 contains {len(eggs1)} eggs. Workbook 2 contains {len(eggs2)} eggs.")
    print(f"Matching egg IDs: {len(common_eggs)}")

    all_abs_data = []
    all_enh_data = []

    for egg_id in common_eggs:
        egg_id = int(egg_id)
        print(f"\nProcessing egg {egg_id}...")

        egg1_data = df1[df1['Egg ID'] == egg_id].reset_index(drop=True)
        egg2_data = df2[df2['Egg ID'] == egg_id].reset_index(drop=True)

        if len(egg1_data) != 20 or len(egg2_data) != 20:
            print(f"  Warning: incomplete data for egg {egg_id} ({len(egg1_data)} vs. {len(egg2_data)} points).")

        abs_rows = []
        enh_rows = []
        abs_colors = []
        enh_colors = []

        for i in range(min(20, len(egg1_data), len(egg2_data))):
            row1 = egg1_data.iloc[i]
            row2 = egg2_data.iloc[i]
            seq = int(row1['Point Number'])

            dr, dg, db = compute_diff(row1, row2, 'abs')
            if dr is not None:
                abs_rows.append({'Point Number': seq, 'Red': dr, 'Green': dg, 'Blue': db})
                abs_colors.append((db, dg, dr))
                all_abs_data.append({'Egg ID': egg_id if i == 0 else '', 'Point Number': seq, 'Red': dr, 'Green': dg, 'Blue': db})
            else:
                abs_rows.append({'Point Number': seq, 'Red': None, 'Green': None, 'Blue': None})
                abs_colors.append(None)
                all_abs_data.append({'Egg ID': egg_id if i == 0 else '', 'Point Number': seq, 'Red': None, 'Green': None, 'Blue': None})

            dr, dg, db = compute_diff(row1, row2, 'enhanced')
            if dr is not None:
                enh_rows.append({'Point Number': seq, 'Red': dr, 'Green': dg, 'Blue': db})
                enh_colors.append((db, dg, dr))
                all_enh_data.append({'Egg ID': egg_id if i == 0 else '', 'Point Number': seq, 'Red': dr, 'Green': dg, 'Blue': db})
            else:
                enh_rows.append({'Point Number': seq, 'Red': None, 'Green': None, 'Blue': None})
                enh_colors.append(None)
                all_enh_data.append({'Egg ID': egg_id if i == 0 else '', 'Point Number': seq, 'Red': None, 'Green': None, 'Blue': None})

        df_abs = pd.DataFrame(abs_rows)
        df_enh = pd.DataFrame(enh_rows)
        df_abs.to_excel(os.path.join(abs_dir, f"egg_{egg_id}.xlsx"), index=False)
        df_enh.to_excel(os.path.join(enh_dir, f"egg_{egg_id}.xlsx"), index=False)

        draw_color_map(abs_colors, os.path.join(abs_dir, f"egg_{egg_id}.png"))
        draw_color_map(enh_colors, os.path.join(enh_dir, f"egg_{egg_id}.png"))

        print(f"  abs/egg_{egg_id}.xlsx, abs/egg_{egg_id}.png")
        print(f"  enhanced/egg_{egg_id}.xlsx, enhanced/egg_{egg_id}.png")

    pd.DataFrame(all_abs_data).to_excel(os.path.join(abs_dir, "summary.xlsx"), index=False)
    pd.DataFrame(all_enh_data).to_excel(os.path.join(enh_dir, "summary.xlsx"), index=False)
    print(f"\nSummary workbooks: abs/summary.xlsx, enhanced/summary.xlsx")

    print("\nDone.")

if __name__ == "__main__":
    main()
