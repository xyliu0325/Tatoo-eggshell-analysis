import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import json
import argparse
from itertools import combinations
from datetime import datetime

def find_egg_region(img):
    """Find the bright egg-shaped region and return its mask, bounds, and tip direction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    egg_contour = max(contours, key=cv2.contourArea)
    egg_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(egg_mask, [egg_contour], -1, 255, -1)

    x, y, w, h = cv2.boundingRect(egg_contour)
    print(f"Egg region: x={x}, y={y}, w={w}, h={h}")

    if len(egg_contour) < 5:
        return egg_mask, (x, y, w, h), 'top', (x + w // 2, y)

    ellipse = cv2.fitEllipse(egg_contour)
    center, axes, angle = ellipse
    cx, cy = center

    contour_points = egg_contour.reshape(-1, 2).astype(np.float64)

    angle_rad = np.deg2rad(angle)

    if axes[0] < axes[1]:
        angle_rad += np.pi / 2

    long_axis_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    centered_points = contour_points - np.array([cx, cy])
    projections = centered_points @ long_axis_dir

    min_proj_idx = projections.argmin()
    max_proj_idx = projections.argmax()

    end1 = contour_points[min_proj_idx]
    end2 = contour_points[max_proj_idx]

    proj_range = projections.max() - projections.min()

    end1_mask = projections < projections.min() + proj_range * 0.2
    end2_mask = projections > projections.max() - proj_range * 0.2

    perp_dir = np.array([-long_axis_dir[1], long_axis_dir[0]])

    if end1_mask.sum() > 2:
        perp_proj1 = centered_points[end1_mask] @ perp_dir
        width1 = perp_proj1.max() - perp_proj1.min()
    else:
        width1 = 999

    if end2_mask.sum() > 2:
        perp_proj2 = centered_points[end2_mask] @ perp_dir
        width2 = perp_proj2.max() - perp_proj2.min()
    else:
        width2 = 999

    if width1 < width2:
        tip_pos = tuple(end1.astype(int))
        tip_angle = angle_rad + np.pi
    else:
        tip_pos = tuple(end2.astype(int))
        tip_angle = angle_rad

    tip_angle = np.arctan2(np.sin(tip_angle), np.cos(tip_angle))

    if -np.pi/4 <= tip_angle < np.pi/4:
        tip_direction = 'right'
    elif np.pi/4 <= tip_angle < 3*np.pi/4:
        tip_direction = 'bottom'
    elif -3*np.pi/4 <= tip_angle < -np.pi/4:
        tip_direction = 'top'
    else:
        tip_direction = 'left'

    print(f"Egg tip: {tip_direction} (end 1 width={width1:.0f}, end 2 width={width2:.0f}, angle={np.rad2deg(tip_angle):.1f} degrees)")

    return egg_mask, (x, y, w, h), tip_direction, tip_pos, tip_angle

def fit_grid(points, tip_direction, n_total=20):
    """
    Fit the grid by sampling candidates and keeping the one with the most matching points.
    Use a 5 x 4 or 4 x 5 grid according to the tip direction.
    """
    if tip_direction in ['top', 'bottom']:
        n_rows, n_cols = 4, 5
    else:
        n_rows, n_cols = 5, 4

    print(f"Grid layout: {n_rows} rows x {n_cols} columns (tip direction: {tip_direction})")
    if len(points) < 8:
        return None, None

    centers = np.array([p['center'] for p in points])
    scores = np.array([p['score'] for p in points])

    top_indices = np.argsort(scores)[-min(20, len(scores)):]
    top_centers = centers[top_indices]

    dists = []
    for i, j in combinations(range(len(top_centers)), 2):
        d = np.linalg.norm(top_centers[i] - top_centers[j])
        if 50 < d < 300:
            dists.append(d)

    if len(dists) < 5:
        return None, None

    dists = np.array(dists)
    hist, bins = np.histogram(dists, bins=50)
    peak_idx = np.argmax(hist)
    base_dist = (bins[peak_idx] + bins[peak_idx + 1]) / 2

    print(f"Estimated grid spacing: {base_dist:.1f}")

    best_grid = None
    best_hits = 0
    best_positions = None

    for start_idx in range(min(len(top_centers), 10)):
        origin = top_centers[start_idx]

        for dx_factor in [0.9, 1.0, 1.1]:
            for dy_factor in [0.9, 1.0, 1.1]:
                dx = base_dist * dx_factor
                dy = base_dist * dy_factor

                positions = []
                for row in range(n_rows):
                    for col in range(n_cols):
                        px = origin[0] + col * dx
                        py = origin[1] + row * dy
                        positions.append((px, py))

                hits = 0
                for cx, cy in top_centers:
                    for px, py in positions:
                        if np.sqrt((cx - px)**2 + (cy - py)**2) < base_dist * 0.4:
                            hits += 1
                            break

                if hits > best_hits:
                    best_hits = hits
                    best_grid = (origin[0], origin[1], dx, dy)
                    best_positions = positions

    min_x, min_y = top_centers.min(axis=0)
    max_x, max_y = top_centers.max(axis=0)
    dx = (max_x - min_x) / (n_cols - 1)
    dy = (max_y - min_y) / (n_rows - 1)

    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            px = min_x + col * dx
            py = min_y + row * dy
            positions.append((px, py))

    hits = 0
    for cx, cy in top_centers:
        for px, py in positions:
            if np.sqrt((cx - px)**2 + (cy - py)**2) < base_dist * 0.4:
                hits += 1
                break

    if hits > best_hits:
        best_hits = hits
        best_grid = (min_x, min_y, dx, dy)
        best_positions = positions

    print(f"Best grid fit: {best_hits}/{len(top_centers)} points matched")
    if best_grid:
        print(f"Grid parameters: origin=({best_grid[0]:.0f}, {best_grid[1]:.0f}), dx={best_grid[2]:.1f}, dy={best_grid[3]:.1f}")

    return best_grid, best_positions

def score_result(dots, predicted_positions):
    """
    Score the detections using three criteria:
    1. Point count, with a preference for counts closer to 20.
    2. Alignment, with a preference for shorter average distances to the grid.
    3. Detection confidence.
    """
    if len(dots) == 0:
        return 0

    count_score = len(dots) * 5

    total_dist = 0
    for dot in dots:
        if 'grid_idx' in dot and dot['grid_idx'] < len(predicted_positions):
            px, py = predicted_positions[dot['grid_idx']]
            cx, cy = dot['center']
            total_dist += np.sqrt((cx - px)**2 + (cy - py)**2)
    avg_dist = total_dist / len(dots) if len(dots) > 0 else 100
    alignment_score = max(0, 100 - avg_dist)

    confidence_score = 0
    for dot in dots:
        if 'score' in dot:
            confidence_score += dot['score'] * 20
        elif not dot.get('forced', False):
            confidence_score += 15

    total_score = count_score + alignment_score + confidence_score
    return total_score

def shift_grid(predicted_positions, direction, n_rows, n_cols):
    """
    Shift the grid by removing a row or column from one side and adding it on the other.
    direction: 'up', 'down', 'left', 'right'
    """
    positions = list(predicted_positions)

    if len(positions) < 2:
        return positions

    dx = positions[1][0] - positions[0][0] if n_cols > 1 else 0
    dy = positions[n_cols][1] - positions[0][1] if n_rows > 1 and len(positions) > n_cols else 0

    if direction == 'up':
        new_positions = []
        for col in range(n_cols):
            old_pos = positions[col]
            new_positions.append((old_pos[0], old_pos[1] - dy))
        new_positions.extend(positions[:-n_cols])
    elif direction == 'down':
        new_positions = positions[n_cols:]
        for col in range(n_cols):
            old_pos = positions[-(n_cols - col)]
            new_positions.append((old_pos[0], old_pos[1] + dy))
    elif direction == 'left':
        new_positions = []
        for row in range(n_rows):
            old_pos = positions[row * n_cols]
            new_positions.append((old_pos[0] - dx, old_pos[1]))
            for col in range(n_cols - 1):
                new_positions.append(positions[row * n_cols + col])
    elif direction == 'right':
        new_positions = []
        for row in range(n_rows):
            for col in range(1, n_cols):
                new_positions.append(positions[row * n_cols + col])
            old_pos = positions[row * n_cols + n_cols - 1]
            new_positions.append((old_pos[0] + dx, old_pos[1]))
    else:
        new_positions = positions

    return new_positions

def assign_dots_to_grid(dots, predicted_positions, max_dist=100):
    """
    Assign at most one point to each grid position.
    Discard points outside the matching distance of every grid position.
    Return the matched points and unmatched grid positions.
    """
    grid_assignments = {}

    for grid_idx, (px, py) in enumerate(predicted_positions):
        best_dot = None
        best_dist = max_dist

        for dot in dots:
            cx, cy = dot['center']
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < best_dist:
                best_dist = dist
                best_dot = dot.copy()
                best_dot['grid_idx'] = grid_idx
                best_dot['grid_dist'] = dist

        if best_dot is not None:
            already_assigned_idx = None
            for assigned_idx, assigned_dot in grid_assignments.items():
                if assigned_dot['center'] == best_dot['center']:
                    already_assigned_idx = assigned_idx
                    break

            if already_assigned_idx is not None:
                if best_dist < grid_assignments[already_assigned_idx]['grid_dist']:
                    del grid_assignments[already_assigned_idx]
                    grid_assignments[grid_idx] = best_dot
            else:
                grid_assignments[grid_idx] = best_dot

    missing_grid_indices = []
    for grid_idx in range(len(predicted_positions)):
        if grid_idx not in grid_assignments:
            missing_grid_indices.append(grid_idx)

    result = []
    for grid_idx in sorted(grid_assignments.keys()):
        dot = grid_assignments[grid_idx]
        dot['grid_matched'] = True
        result.append(dot)

    return result, missing_grid_indices

def nms_dots(dots, min_dist=50):
    """Remove duplicate detections using non-maximum suppression."""
    if len(dots) == 0:
        return []

    dots_sorted = sorted(dots, key=lambda d: d['score'], reverse=True)

    keep = []
    for dot in dots_sorted:
        cx, cy = dot['center']
        is_duplicate = False
        for kept in keep:
            kx, ky = kept['center']
            dist = np.sqrt((cx - kx)**2 + (cy - ky)**2)
            if dist < min_dist:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(dot)

    return keep


MANUAL_SAMPLE_RADIUS = 22

MANUAL_DISPLAY_RADIUS = 16
MANUAL_SELECTED_RADIUS = 19
MANUAL_LABEL_FONT_SCALE = 1.05
MANUAL_LABEL_OFFSET = 12


def make_circle_contour(center, radius=MANUAL_SAMPLE_RADIUS, n_points=72):
    """
    Build a fixed circular contour around a center point.
    Replace the SAM contour after manual adjustment so oversized or incorrect masks do not affect color sampling.
    """
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ], axis=1)
    return pts.reshape(-1, 1, 2).astype(np.int32)


def force_dot_to_manual_circle(dot, radius=MANUAL_SAMPLE_RADIUS):
    """
    Replace a point's contour with a fixed circular ROI centered on the point.
    The color sampler uses this circular region.
    """
    cx, cy = dot['center']
    dot['center'] = (int(round(cx)), int(round(cy)))
    dot['contour'] = make_circle_contour(dot['center'], radius=radius)
    dot['area'] = int(np.pi * radius * radius)
    dot['manual_fixed_roi'] = True
    dot['sample_radius'] = int(radius)
    return dot


def set_candidate_xy(dot, new_x, new_y):
    """
    Update the point center and replace its contour with a fixed circular ROI.
    A translated SAM contour could still contain segmentation errors,
    which can affect color sampling and draw large stray outlines in the output.
    """
    dot['center'] = (int(round(new_x)), int(round(new_y)))
    return force_dot_to_manual_circle(dot, radius=dot.get('sample_radius', MANUAL_SAMPLE_RADIUS))

def manual_adjust_dot_candidates(dot_candidates, img, window_name='Manual adjust'):
    """
    Manually adjust the 20 points after numbering in Step 6.

    Controls:
    - Left-click and drag to move the nearest point.
    - Press n or Tab to select the next point.
    - Press p to select the previous point.
    - Use the arrow keys to nudge the current point.
    - Press + or - to change the nudge step.
    - Press Enter to confirm the image and sample RGB at the adjusted points.
    - Press q or Esc to skip the current image.
    """
    if dot_candidates is None or len(dot_candidates) == 0:
        print("No points available for adjustment.")
        return dot_candidates, False

    original = img.copy()
    h, w = original.shape[:2]

    max_w, max_h = 1500, 950
    scale = min(max_w / w, max_h / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)

    state = {
        'selected': 0,
        'dragging': False,
        'step': 2,
        'skip': False
    }

    def get_xy(dot):
        return float(dot['center'][0]), float(dot['center'][1])

    def nearest_point(px, py):
        dists = []
        for idx, dot in enumerate(dot_candidates):
            x, y = get_xy(dot)
            dists.append((idx, (x - px) ** 2 + (y - py) ** 2))
        return min(dists, key=lambda t: t[1])[0]

    def draw():
        if scale < 1.0:
            canvas = cv2.resize(original, (disp_w, disp_h))
        else:
            canvas = original.copy()

        for idx, dot in enumerate(dot_candidates):
            x, y = get_xy(dot)
            sx, sy = int(round(x * scale)), int(round(y * scale))

            color = (0, 255, 0)
            circle_radius = MANUAL_DISPLAY_RADIUS
            thickness = 3
            if idx == state['selected']:
                color = (0, 0, 255)
                circle_radius = MANUAL_SELECTED_RADIUS
                thickness = 4

            cv2.circle(canvas, (sx, sy), circle_radius, color, thickness)
            cv2.circle(canvas, (sx, sy), 4, color, -1)

            label = str(idx + 1)
            label_pos = (sx + MANUAL_LABEL_OFFSET, sy - MANUAL_LABEL_OFFSET)

            cv2.putText(
                canvas,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                MANUAL_LABEL_FONT_SCALE,
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )
            cv2.putText(
                canvas,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                MANUAL_LABEL_FONT_SCALE,
                color,
                2,
                cv2.LINE_AA
            )

        info_lines = [
            "Drag dot with mouse | Arrow keys: fine adjust | n/Tab: next | p: previous",
            "+/-: change step | Enter: confirm and sample RGB | q/Esc: skip current image",
            f"Selected point: {state['selected'] + 1}/{len(dot_candidates)} | step: {state['step']} px"
        ]

        y0 = 36
        for line in info_lines:
            cv2.putText(canvas, line, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            y0 += 34

        return canvas

    def mouse_callback(event, mx, my, flags, param):
        px = mx / scale
        py = my / scale

        if event == cv2.EVENT_LBUTTONDOWN:
            state['selected'] = nearest_point(px, py)
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], px, py)
            state['dragging'] = True

        elif event == cv2.EVENT_MOUSEMOVE and state['dragging']:
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], px, py)

        elif event == cv2.EVENT_LBUTTONUP:
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], px, py)
            state['dragging'] = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, draw())
        key = cv2.waitKeyEx(30)

        if key in [13, 10]:
            break

        elif key in [27, ord('q'), ord('Q')]:
            state['skip'] = True
            break

        elif key in [ord('n'), ord('N'), 9]:
            state['selected'] = (state['selected'] + 1) % len(dot_candidates)

        elif key in [ord('p'), ord('P')]:
            state['selected'] = (state['selected'] - 1) % len(dot_candidates)

        elif key in [ord('+'), ord('=')]:
            state['step'] = min(50, state['step'] + 1)

        elif key in [ord('-'), ord('_')]:
            state['step'] = max(1, state['step'] - 1)

        elif key in [2424832, 81]:
            x, y = get_xy(dot_candidates[state['selected']])
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], x - state['step'], y)

        elif key in [2555904, 83]:
            x, y = get_xy(dot_candidates[state['selected']])
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], x + state['step'], y)

        elif key in [2490368, 82]:
            x, y = get_xy(dot_candidates[state['selected']])
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], x, y - state['step'])

        elif key in [2621440, 84]:
            x, y = get_xy(dot_candidates[state['selected']])
            dot_candidates[state['selected']] = set_candidate_xy(dot_candidates[state['selected']], x, y + state['step'])

    cv2.destroyWindow(window_name)

    if not state['skip']:
        for idx in range(len(dot_candidates)):
            dot_candidates[idx] = force_dot_to_manual_circle(dot_candidates[idx], radius=MANUAL_SAMPLE_RADIUS)

    return dot_candidates, not state['skip']


def detect_dots_sam(image_path, model_path="sam_vit_b.pth", erode_ratio=0.3, manual_adjust=True):
    if not os.path.exists(model_path):
        print("Downloading the SAM checkpoint...")
        import urllib.request
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            model_path
        )

    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    print("\n=== Step 1: Locate the egg region ===")
    result = find_egg_region(img)
    egg_mask, egg_rect, tip_direction, tip_pos, tip_angle = result

    if egg_mask is None:
        print("Could not locate the egg region.")
        return

    x, y, w, h = egg_rect
    padding = 10
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = egg_mask[y1:y2, x1:x2]

    print("\n=== Step 2: Run SAM detection ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=48,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=150,
    )

    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    print("Generating masks...")
    masks = mask_generator.generate(cropped_rgb)
    print(f"SAM generated {len(masks)} masks.")

    print("\n=== Step 3: Filter initial detections ===")
    dot_candidates = []

    for mask_data in masks:
        mask = mask_data['segmentation']
        area = mask_data['area']

        if area < 400 or area > 25000:
            continue

        M_temp = cv2.moments(mask.astype(np.uint8))
        if M_temp["m00"] > 0:
            cx_temp = int(M_temp["m10"] / M_temp["m00"])
            cy_temp = int(M_temp["m01"] / M_temp["m00"])
            if cropped_mask[cy_temp, cx_temp] == 0:
                continue

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            cnt_orig = cnt + np.array([x1, y1])

            dot_candidates.append({
                'center': (cx, cy),
                'contour': cnt_orig,
                'area': area,
                'circularity': circularity,
                'score': mask_data.get('stability_score', 0.9) * circularity
            })

    print(f"Initial candidates: {len(dot_candidates)} points")

    all_initial_candidates = [d.copy() for d in dot_candidates]

    print("\n=== Step 4: Fit the grid ===")
    grid_params, predicted_positions = fit_grid(dot_candidates, tip_direction)

    grid_basis_dots = []
    if grid_params and len(dot_candidates) >= 8:
        scores = [d['score'] for d in dot_candidates]
        top_indices = np.argsort(scores)[-min(16, len(scores)):]
        grid_basis_dots = [dot_candidates[i] for i in top_indices]

    if predicted_positions:
        print(f"Predicted {len(predicted_positions)} grid positions.")

        if len(predicted_positions) >= 2:
            p1 = np.array(predicted_positions[0])
            p2 = np.array(predicted_positions[1])
            grid_spacing = np.linalg.norm(p2 - p1)
            max_dist = grid_spacing * 0.6
        else:
            max_dist = 100

        dot_candidates, missing_indices = assign_dots_to_grid(dot_candidates, predicted_positions, max_dist)
        print(f"Grid match: {len(dot_candidates)}/20 positions have a point.")

        dot_candidates_backup = [d.copy() for d in dot_candidates]

        if len(dot_candidates) >= 5:
            offset_x = 0
            offset_y = 0
            count = 0
            for dot in dot_candidates:
                grid_idx = dot['grid_idx']
                px, py = predicted_positions[grid_idx]
                cx, cy = dot['center']
                offset_x += cx - px
                offset_y += cy - py
                count += 1

            if count > 0:
                offset_x /= count
                offset_y /= count
                print(f"Grid offset correction: dx={offset_x:.1f}, dy={offset_y:.1f}")

                predicted_positions = [(px + offset_x, py + offset_y) for px, py in predicted_positions]

                dot_candidates, missing_indices = assign_dots_to_grid(dot_candidates, predicted_positions, max_dist)
                print(f"After correction: {len(dot_candidates)}/20 positions have a point.")

        predictor = SamPredictor(sam)
        predictor.set_image(cropped_rgb)

        if missing_indices:
            print(f"Searching {len(missing_indices)} missing positions with point prompts...")

            for grid_idx in missing_indices:
                px, py = predicted_positions[grid_idx]
                local_x = px - x1
                local_y = py - y1

                if 0 <= local_x < cropped_img.shape[1] and 0 <= local_y < cropped_img.shape[0]:
                    input_point = np.array([[local_x, local_y]])
                    input_label = np.array([1])

                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                    )

                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]

                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(cnt)

                        if area > 100:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"]) + x1
                                cy = int(M["m01"] / M["m00"]) + y1
                                cnt_orig = cnt + np.array([x1, y1])

                                dot_candidates.append({
                                    'center': (cx, cy),
                                    'contour': cnt_orig,
                                    'area': area,
                                    'grid_idx': grid_idx,
                                    'grid_matched': True,
                                    'forced': True
                                })
                                print(f"  Grid position {grid_idx}: found a point at ({cx}, {cy})")
                        else:
                            print(f"  Grid position {grid_idx}: mask too small. Skipping.")
                    else:
                        print(f"  Grid position {grid_idx}: no contour found.")

            print(f"After filling gaps: {len(dot_candidates)}/20 points")

        print("\n=== Step 4.5: Refine the grid iteratively ===")

        if tip_direction in ['top', 'bottom']:
            n_rows, n_cols = 4, 5
        else:
            n_rows, n_cols = 5, 4

        best_score = score_result(dot_candidates, predicted_positions)
        best_dots = dot_candidates
        best_positions = predicted_positions
        print(f"Initial score: {best_score:.1f}")

        max_iterations = 5
        for iteration in range(max_iterations):
            improved = False

            for direction in ['up', 'down', 'left', 'right']:
                shifted_positions = shift_grid(best_positions, direction, n_rows, n_cols)

                shifted_dots, shifted_missing = assign_dots_to_grid(
                    [d.copy() for d in all_initial_candidates], shifted_positions, max_dist
                )

                if shifted_missing and predictor is not None:
                    for grid_idx in shifted_missing:
                        px, py = shifted_positions[grid_idx]
                        local_x = px - x1
                        local_y = py - y1

                        if 0 <= local_x < cropped_img.shape[1] and 0 <= local_y < cropped_img.shape[0]:
                            input_point = np.array([[local_x, local_y]])
                            input_label = np.array([1])

                            try:
                                masks_pred, scores_pred, _ = predictor.predict(
                                    point_coords=input_point,
                                    point_labels=input_label,
                                    multimask_output=True
                                )

                                best_mask_idx = np.argmax(scores_pred)
                                mask = masks_pred[best_mask_idx]

                                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    cnt = max(contours, key=cv2.contourArea)
                                    area = cv2.contourArea(cnt)

                                    if area > 100:
                                        M = cv2.moments(cnt)
                                        if M["m00"] > 0:
                                            cx = int(M["m10"] / M["m00"]) + x1
                                            cy = int(M["m01"] / M["m00"]) + y1
                                            cnt_orig = cnt + np.array([x1, y1])

                                            shifted_dots.append({
                                                'center': (cx, cy),
                                                'contour': cnt_orig,
                                                'area': area,
                                                'grid_idx': grid_idx,
                                                'grid_matched': True,
                                                'forced': True
                                            })
                            except:
                                pass

                shifted_score = score_result(shifted_dots, shifted_positions)

                if shifted_score > best_score:
                    print(f"  Iteration {iteration+1}, {direction}: score {shifted_score:.1f} exceeds {best_score:.1f}. Updating the grid.")
                    best_score = shifted_score
                    best_dots = shifted_dots
                    best_positions = shifted_positions
                    improved = True
                    break

            if not improved:
                print(f"  Iteration {iteration+1}: no improvement. Stopping.")
                break

        dot_candidates = best_dots
        predicted_positions = best_positions
        print(f"Refined score: {best_score:.1f}, points: {len(dot_candidates)}")

        print("\n=== Step 4.6: Refine individual points ===")

        for i, dot in enumerate(dot_candidates):
            if 'grid_idx' not in dot:
                continue

            grid_idx = dot['grid_idx']
            if grid_idx >= len(predicted_positions):
                continue

            px, py = predicted_positions[grid_idx]
            cx, cy = dot['center']

            current_dist = np.sqrt((cx - px)**2 + (cy - py)**2)

            local_x = px - x1
            local_y = py - y1

            if 0 <= local_x < cropped_img.shape[1] and 0 <= local_y < cropped_img.shape[0]:
                try:
                    input_point = np.array([[local_x, local_y]])
                    input_label = np.array([1])

                    masks_pred, scores_pred, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                    )

                    best_mask_idx = np.argmax(scores_pred)
                    mask = masks_pred[best_mask_idx]

                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(cnt)

                        if area > 100:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                new_cx = int(M["m10"] / M["m00"]) + x1
                                new_cy = int(M["m01"] / M["m00"]) + y1

                                new_dist = np.sqrt((new_cx - px)**2 + (new_cy - py)**2)

                                if new_dist < current_dist * 0.8:
                                    cnt_orig = cnt + np.array([x1, y1])
                                    dot_candidates[i] = {
                                        'center': (new_cx, new_cy),
                                        'contour': cnt_orig,
                                        'area': area,
                                        'grid_idx': grid_idx,
                                        'grid_matched': True,
                                        'forced': True,
                                        'optimized': True
                                    }
                                    print(f"  Point {i+1}: distance reduced from {current_dist:.1f} to {new_dist:.1f}")
                except:
                    pass

        print("Point refinement complete.")

        print("\n=== Step 4.7: Check for overlapping detections ===")

        def is_center_inside_contour(dot_small, cnt_large):
            """Check whether the smaller point's center is inside the larger contour."""
            cx, cy = dot_small['center']
            result = cv2.pointPolygonTest(cnt_large, (float(cx), float(cy)), False)
            return result > 0

        def get_contour_area(dot):
            if 'contour' in dot:
                return cv2.contourArea(dot['contour'])
            return dot.get('area', 0)

        def try_point_prompt_at_position(predictor, local_x, local_y, x1, y1,
                                          negative_points=None, expected_area_range=(100, 15000)):
            """Try a point prompt and return (center, contour, area, score) on success."""
            input_points = [[local_x, local_y]]
            input_labels = [1]

            if negative_points:
                for nx, ny in negative_points:
                    input_points.append([nx, ny])
                    input_labels.append(0)

            input_point = np.array(input_points)
            input_label = np.array(input_labels)

            masks_pred, scores_pred, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )

            best_result = None
            best_score = -1

            for mask_idx in range(len(masks_pred)):
                mask = masks_pred[mask_idx]
                area = mask.sum()
                score = scores_pred[mask_idx]

                if expected_area_range[0] < area < expected_area_range[1]:
                    if score > best_score:
                        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            cnt_area = cv2.contourArea(cnt)
                            if cnt_area > 100:
                                M = cv2.moments(cnt)
                                if M["m00"] > 0:
                                    cx = int(M["m10"] / M["m00"]) + x1
                                    cy = int(M["m01"] / M["m00"]) + y1
                                    cnt_orig = cnt + np.array([x1, y1])
                                    best_score = score
                                    best_result = {
                                        'center': (cx, cy),
                                        'contour': cnt_orig,
                                        'area': cnt_area,
                                        'score': score
                                    }

            return best_result

        contained_indices = set()
        container_map = {}
        for i, dot1 in enumerate(dot_candidates):
            for j, dot2 in enumerate(dot_candidates):
                if i == j:
                    continue
                if 'contour' not in dot2:
                    continue

                if is_center_inside_contour(dot1, dot2['contour']):
                    contained_indices.add(i)
                    container_map[i] = j
                    print(f"  The center of point {i+1} is inside the contour of point {j+1}.")

        if len(predicted_positions) >= 2:
            p1 = np.array(predicted_positions[0])
            p2 = np.array(predicted_positions[1])
            search_step = np.linalg.norm(p2 - p1) * 0.3
        else:
            search_step = 30

        for i in contained_indices:
            dot = dot_candidates[i]
            if 'grid_idx' not in dot:
                continue

            grid_idx = dot['grid_idx']
            if grid_idx >= len(predicted_positions):
                continue

            px, py = predicted_positions[grid_idx]

            negative_points = []
            if i in container_map:
                j = container_map[i]
                dot2 = dot_candidates[j]
                cx2, cy2 = dot2['center']
                negative_points.append([cx2 - x1, cy2 - y1])

            container_area = get_contour_area(dot_candidates[container_map[i]]) if i in container_map else 10000
            expected_area_range = (100, container_area * 0.8)

            print(f"  Point {i+1}: retrying segmentation (grid position={grid_idx}, target area<{container_area*0.8:.0f})")

            local_x = px - x1
            local_y = py - y1

            best_result = None
            best_score = -1
            best_direction = "original position"

            if 0 <= local_x < cropped_img.shape[1] and 0 <= local_y < cropped_img.shape[0]:
                try:
                    result = try_point_prompt_at_position(
                        predictor, local_x, local_y, x1, y1,
                        negative_points, expected_area_range
                    )
                    if result:
                        still_contained = False
                        if i in container_map:
                            j = container_map[i]
                            if 'contour' in dot_candidates[j]:
                                new_cx, new_cy = result['center']
                                test_result = cv2.pointPolygonTest(
                                    dot_candidates[j]['contour'],
                                    (float(new_cx), float(new_cy)), False
                                )
                                still_contained = test_result > 0

                        if not still_contained:
                            best_result = result
                            best_score = result['score']
                            print(f"    Original position: score={result['score']:.3f}, no longer inside another contour.")
                        else:
                            print(f"    Original position: score={result['score']:.3f}, still inside another contour. Searching nearby positions.")
                except Exception as e:
                    print(f"    Segmentation at the original position failed: {e}")

            if best_result is None or best_score < 0.5:
                directions = [
                    ('up', 0, -search_step),
                    ('down', 0, search_step),
                    ('left', -search_step, 0),
                    ('right', search_step, 0),
                    ('upper left', -search_step, -search_step),
                    ('upper right', search_step, -search_step),
                    ('lower left', -search_step, search_step),
                    ('lower right', search_step, search_step),
                ]

                for dir_name, dx, dy in directions:
                    test_x = local_x + dx
                    test_y = local_y + dy

                    if 0 <= test_x < cropped_img.shape[1] and 0 <= test_y < cropped_img.shape[0]:
                        try:
                            result = try_point_prompt_at_position(
                                predictor, test_x, test_y, x1, y1,
                                negative_points, expected_area_range
                            )
                            if result and result['score'] > best_score:
                                still_contained = False
                                if i in container_map:
                                    j = container_map[i]
                                    if 'contour' in dot_candidates[j]:
                                        new_cx, new_cy = result['center']
                                        test_result = cv2.pointPolygonTest(
                                            dot_candidates[j]['contour'],
                                            (float(new_cx), float(new_cy)), False
                                        )
                                        still_contained = test_result > 0

                                if not still_contained:
                                    best_result = result
                                    best_score = result['score']
                                    best_direction = dir_name
                                    print(f"    {dir_name}: score={result['score']:.3f}, no longer inside another contour.")
                        except:
                            pass

            if best_result is not None:
                dot_candidates[i] = {
                    'center': best_result['center'],
                    'contour': best_result['contour'],
                    'area': best_result['area'],
                    'grid_idx': grid_idx,
                    'grid_matched': True,
                    'forced': True,
                    'fixed_containment': True,
                    'search_direction': best_direction
                }
                print(f"  Point {i+1}: segmentation refined ({best_direction}, score={best_score:.3f})")
            else:
                print(f"  Point {i+1}: segmentation refinement failed. Keeping the original.")
    else:
        print("Grid fitting failed.")
        predicted_positions = []

    print("\n=== Step 5: Review grid matches ===")
    print(f"Final count: {len(dot_candidates)} points")

    print(f"\n=== Step 6: Number points by tip direction ({tip_direction}) ===")

    if tip_direction in ['top', 'bottom']:
        n_rows, n_cols = 4, 5
    else:
        n_rows, n_cols = 5, 4


    def get_display_order(grid_idx, tip_dir, n_rows, n_cols):
        """Use the tip direction to calculate point order.
        With the tip facing up, number points from left to right across each row.
        """
        row = grid_idx // n_cols
        col = grid_idx % n_cols

        if tip_dir == 'top':
            return row * n_cols + col
        elif tip_dir == 'bottom':
            new_row = (n_rows - 1) - row
            new_col = (n_cols - 1) - col
            return new_row * n_cols + new_col
        elif tip_dir == 'left':
            new_row = col
            new_col = (n_rows - 1) - row
            return new_row * n_rows + new_col
        else:
            new_row = (n_cols - 1) - col
            new_col = row
            return new_row * n_rows + new_col

    for dot in dot_candidates:
        if 'grid_idx' in dot:
            dot['display_order'] = get_display_order(dot['grid_idx'], tip_direction, n_rows, n_cols)
        else:
            dot['display_order'] = 999

    dot_candidates.sort(key=lambda d: d.get('display_order', 999))

    if len(dot_candidates) != 20:
        print(f"Warning: found {len(dot_candidates)} points. Continuing with adjustment and color sampling (expected 20).")

    if manual_adjust:
        dot_candidates, keep_current = manual_adjust_dot_candidates(
            dot_candidates,
            img,
            window_name=f"Adjust: {os.path.basename(image_path)}"
        )

        if not keep_current:
            print("User skipped the current image.")
            return None

    output_base = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    cache_file = os.path.join(output_dir, f"{output_base}_cache.json")

    cache_data = {
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(),
        "tip_direction": tip_direction,
        "tip_pos": [int(tip_pos[0]), int(tip_pos[1])],
        "tip_angle": float(tip_angle),
        "egg_rect": [int(x), int(y), int(w), int(h)],
        "dots": []
    }

    for i, dot in enumerate(dot_candidates):
        dot_data = {
            "index": i + 1,
            "center": [int(dot['center'][0]), int(dot['center'][1])],
            "area": int(dot.get('area', 0)),
            "grid_idx": int(dot.get('grid_idx', -1)),
            "display_order": int(dot.get('display_order', 999)),
            "contour": dot['contour'].tolist() if 'contour' in dot else [],
            "manual_fixed_roi": bool(dot.get("manual_fixed_roi", False)),
            "sample_radius": int(dot.get("sample_radius", MANUAL_SAMPLE_RADIUS))
        }
        cache_data["dots"].append(dot_data)

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"\nDetection cache saved to: {cache_file}")

    dot_candidates, img, egg_mask = do_color_sampling(dot_candidates, img, egg_mask, erode_ratio)

    save_results(dot_candidates, img, egg_mask, image_path, output_dir, output_base)

    return dot_candidates


def do_color_sampling(dot_candidates, img, egg_mask, erode_ratio=0.3):
    """Sample color by growing a region from the center based on color similarity."""
    print("\n=== Step 7: Sample point colors ===")

    def get_fixed_circle_sample_region(img, center, radius=MANUAL_SAMPLE_RADIUS):
        """
        For manually adjusted points, sample only inside a fixed circle
        centered at the user-confirmed position.
        """
        cx, cy = int(round(center[0])), int(round(center[1]))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(radius), 255, -1)

        pixels = img[mask > 0]
        if len(pixels) > 0:
            representative_color = np.median(pixels, axis=0).astype(np.uint8)
        else:
            representative_color = img[max(0, min(cy, h - 1)), max(0, min(cx, w - 1))].astype(np.uint8)

        circle_cnt = make_circle_contour((cx, cy), radius=radius)
        return circle_cnt, representative_color, int(mask.sum() / 255)

    def get_color_sample_region(img, contour, center, erode_ratio=0.3):
        """
        Grow the sampling region outward from the center using color similarity.
        """
        cx, cy = center
        x, y, w, h = cv2.boundingRect(contour)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_contour = contour - np.array([x1, y1])
        cv2.drawContours(local_mask, [shifted_contour], -1, 255, -1)

        local_img = img[y1:y2, x1:x2]
        local_cx, local_cy = cx - x1, cy - y1

        local_cx = max(0, min(local_cx, local_img.shape[1] - 1))
        local_cy = max(0, min(local_cy, local_img.shape[0] - 1))

        seed_region = local_img[
            max(0, local_cy-2):min(local_img.shape[0], local_cy+3),
            max(0, local_cx-2):min(local_img.shape[1], local_cx+3)
        ]
        if seed_region.size > 0:
            seed_color = np.median(seed_region.reshape(-1, 3), axis=0)
        else:
            seed_color = local_img[local_cy, local_cx].astype(np.float32)

        color_diff = np.sqrt(np.sum((local_img.astype(np.float32) - seed_color) ** 2, axis=2))

        yy, xx = np.ogrid[:local_img.shape[0], :local_img.shape[1]]
        dist_from_center = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)

        max_color_diff = 100
        color_score = np.clip(1 - color_diff / max_color_diff, 0, 1)

        max_dist = np.sqrt(w**2 + h**2) / 2
        dist_score = np.clip(1 - dist_from_center / max_dist, 0, 1)

        combined_score = color_score * 0.7 + dist_score * 0.3

        combined_score[local_mask == 0] = 0

        threshold = 0.5
        sample_mask = (combined_score > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sample_mask = cv2.morphologyEx(sample_mask, cv2.MORPH_OPEN, kernel)
        sample_mask = cv2.morphologyEx(sample_mask, cv2.MORPH_CLOSE, kernel)

        contours_sample, _ = cv2.findContours(sample_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0
        for cnt in contours_sample:
            if cv2.pointPolygonTest(cnt, (local_cx, local_cy), False) >= 0:
                area = cv2.contourArea(cnt)
                if area > best_area:
                    best_area = area
                    best_cnt = cnt

        if best_cnt is None and contours_sample:
            best_cnt = max(contours_sample, key=cv2.contourArea)
            best_area = cv2.contourArea(best_cnt)

        if best_area < 100:
            threshold = 0.3
            sample_mask = (combined_score > threshold).astype(np.uint8) * 255
            sample_mask = cv2.morphologyEx(sample_mask, cv2.MORPH_OPEN, kernel)
            sample_mask = cv2.morphologyEx(sample_mask, cv2.MORPH_CLOSE, kernel)

            contours_sample, _ = cv2.findContours(sample_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_sample:
                for cnt in contours_sample:
                    if cv2.pointPolygonTest(cnt, (local_cx, local_cy), False) >= 0:
                        area = cv2.contourArea(cnt)
                        if area > best_area:
                            best_area = area
                            best_cnt = cnt
                if best_cnt is None:
                    best_cnt = max(contours_sample, key=cv2.contourArea)
                    best_area = cv2.contourArea(best_cnt)

        contour_area = cv2.contourArea(contour)
        min_sample_area = contour_area / 3

        if best_cnt is not None and best_area > 30 and best_area >= min_sample_area:
            final_mask = np.zeros_like(sample_mask)
            cv2.drawContours(final_mask, [best_cnt], -1, 255, -1)

            final_cnt_orig = best_cnt + np.array([x1, y1])

            final_pixels = local_img[final_mask > 0]
            if len(final_pixels) > 0:
                representative_color = np.median(final_pixels, axis=0).astype(np.uint8)
            else:
                representative_color = seed_color.astype(np.uint8)

            return final_cnt_orig, representative_color, best_area

        contour_area = cv2.contourArea(contour)
        for ratio in [erode_ratio, erode_ratio * 0.5, erode_ratio * 0.25, 0]:
            if ratio == 0:
                eroded_mask = local_mask.copy()
            else:
                kernel_size = max(3, int(np.sqrt(contour_area) * ratio))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                eroded_mask = cv2.erode(local_mask, kernel, iterations=1)

            eroded_area = eroded_mask.sum() / 255
            if eroded_area >= min_sample_area:
                break

        eroded_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if eroded_contours:
            eroded_cnt = max(eroded_contours, key=cv2.contourArea)
            eroded_cnt_orig = eroded_cnt + np.array([x1, y1])
        else:
            eroded_cnt_orig = contour

        pixels = local_img[eroded_mask > 0]
        representative_color = np.median(pixels, axis=0).astype(np.uint8) if len(pixels) > 0 else seed_color.astype(np.uint8)

        return eroded_cnt_orig, representative_color, eroded_mask.sum()

    for i, dot in enumerate(dot_candidates):
        if 'contour' not in dot:
            continue

        center = dot['center']

        if dot.get('manual_fixed_roi', False):
            eroded_cnt, rep_color, sample_area = get_fixed_circle_sample_region(
                img, center, radius=dot.get('sample_radius', MANUAL_SAMPLE_RADIUS)
            )
        else:
            eroded_cnt, rep_color, sample_area = get_color_sample_region(img, dot['contour'], center, erode_ratio)

        dot['color_sample_contour'] = eroded_cnt
        dot['representative_color'] = rep_color
        dot['sample_area'] = sample_area

        r, g, b = int(rep_color[2]), int(rep_color[1]), int(rep_color[0])
        print(f"  Point {i+1}: RGB=({r}, {g}, {b}), sampling area={sample_area}")

    return dot_candidates, img, egg_mask


def save_results(dot_candidates, img, egg_mask, image_path, output_dir, output_base):
    """Save the annotated image and JSON results."""
    result = img.copy()

    egg_contours, _ = cv2.findContours(egg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, egg_contours, -1, (255, 255, 0), 2)

    for i, dot in enumerate(dot_candidates):
        cx, cy = dot['center']
        cnt = dot['contour']

        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

        if 'color_sample_contour' in dot:
            cv2.drawContours(result, [dot['color_sample_contour']], -1, (0, 255, 255), 2)

            M = cv2.moments(dot['color_sample_contour'])
            if M["m00"] > 0:
                sample_cx = int(M["m10"] / M["m00"])
                sample_cy = int(M["m01"] / M["m00"])
                cv2.circle(result, (sample_cx, sample_cy), 6, (255, 0, 0), -1)

        cv2.circle(result, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(result, str(i+1), (cx+16, cy+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(result, str(i+1), (cx+16, cy+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    result_img_path = os.path.join(output_dir, f"{output_base}_result.jpg")
    cv2.imwrite(result_img_path, result)
    print(f"\nResult image saved to: {result_img_path}")

    final_result = {
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(),
        "dot_count": len(dot_candidates),
        "dots": []
    }

    for i, dot in enumerate(dot_candidates):
        dot_data = {
            "index": i + 1,
            "center": [int(dot['center'][0]), int(dot['center'][1])],
            "area": int(dot.get('area', 0)),
        }
        if 'representative_color' in dot:
            bgr = dot['representative_color']
            dot_data["color_rgb"] = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
            dot_data["color_bgr"] = [int(bgr[0]), int(bgr[1]), int(bgr[2])]
        if 'sample_area' in dot:
            dot_data["sample_area"] = int(dot['sample_area'])
        if 'color_sample_contour' in dot:
            dot_data["sample_contour"] = dot['color_sample_contour'].tolist()
        final_result["dots"].append(dot_data)

    result_json_path = os.path.join(output_dir, f"{output_base}_result.json")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    print(f"Result JSON saved to: {result_json_path}")

    print("\n=== Color summary ===")
    for i, dot in enumerate(dot_candidates):
        if 'representative_color' in dot:
            bgr = dot['representative_color']
            print(f"Point {i+1}: RGB({bgr[2]}, {bgr[1]}, {bgr[0]})")

    print(f"\nDetected {len(dot_candidates)} points.")


def load_from_cache(cache_file, image_path):
    """Load cached detection results."""
    print(f"Loading cache: {cache_file}")

    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    egg_contour = max(contours, key=cv2.contourArea)
    egg_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(egg_mask, [egg_contour], -1, 255, -1)

    dot_candidates = []
    for dot_data in cache_data["dots"]:
        dot = {
            "center": tuple(dot_data["center"]),
            "area": dot_data["area"],
            "grid_idx": dot_data.get("grid_idx", -1),
            "display_order": dot_data.get("display_order", 999),
            "contour": np.array(dot_data["contour"], dtype=np.int32),
            "manual_fixed_roi": bool(dot_data.get("manual_fixed_roi", False)),
            "sample_radius": int(dot_data.get("sample_radius", MANUAL_SAMPLE_RADIUS))
        }
        dot_candidates.append(dot)

    print(f"Loaded {len(dot_candidates)} points.")
    return dot_candidates, img, egg_mask, cache_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect dye dots with SAM and sample their colors.')
    parser.add_argument('image', nargs='?', default="src/good/30c25a83f25208a487b340998aea22b6.jpg",
                        help='Input image path')
    parser.add_argument('--from-cache', type=str, default=None,
                        help='Load detections from a JSON cache and go directly to color sampling')
    parser.add_argument('--erode-ratio', type=float, default=0.3,
                        help='Sampling region erosion ratio. Larger values shrink the region more (default: 0.3)')
    parser.add_argument('--manual-adjust', dest='manual_adjust', action='store_true', default=True,
                        help='Enable the manual adjustment window (default)')
    parser.add_argument('--no-manual-adjust', dest='manual_adjust', action='store_false',
                        help='Disable the manual adjustment window')

    args = parser.parse_args()

    if args.from_cache:
        dot_candidates, img, egg_mask, cache_data = load_from_cache(args.from_cache, args.image)
        if dot_candidates is None:
            sys.exit(1)

        output_base = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        if args.manual_adjust:
            dot_candidates, keep_current = manual_adjust_dot_candidates(
                dot_candidates,
                img,
                window_name=f"Adjust: {os.path.basename(args.image)}"
            )
            if not keep_current:
                print("User skipped the current image.")
                sys.exit(0)

        dot_candidates, img, egg_mask = do_color_sampling(dot_candidates, img, egg_mask, args.erode_ratio)
        save_results(dot_candidates, img, egg_mask, args.image, output_dir, output_base)
    else:
        detect_dots_sam(args.image, erode_ratio=args.erode_ratio, manual_adjust=args.manual_adjust)
