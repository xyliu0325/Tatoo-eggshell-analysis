import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import re
import argparse

from detect_dots_sam_v4_lxy import (
    detect_dots_sam,
    do_color_sampling,
    save_results,
    load_from_cache,
    manual_adjust_dot_candidates
)


def natural_key(path):
    """
    Sort filenames such as 1.jpg, 2.jpg, and 10.jpg
    in numeric order.
    """
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


parser = argparse.ArgumentParser(description="Detect 20 dye dots per egg and sample their RGB colors.")
parser.add_argument('--from-cache', action='store_true', help='Load cached detections for manual adjustment and color sampling')
parser.add_argument('--erode-ratio', type=float, default=0.3, help='Sampling region erosion ratio')
parser.add_argument('--no-manual-adjust', action='store_true', help='Disable the manual adjustment window')
args = parser.parse_args()

manual_adjust = not args.no_manual_adjust


good_images = sorted(glob.glob("src/good/*.jpg"), key=natural_key)
bad_images = sorted(glob.glob("src/bad/*.jpg"), key=natural_key)

all_images = good_images + bad_images


print(f"Found {len(all_images)} images.")
print(f"Mode: {'Cached detections and color sampling' if args.from_cache else 'Full detection'}")
print(f"Manual adjustment: {'on' if manual_adjust else 'off'}")
print("=" * 60)

os.makedirs("results", exist_ok=True)


for i, img_path in enumerate(all_images):
    print(f"\n[{i + 1}/{len(all_images)}] Processing: {img_path}")
    print("-" * 60)

    output_base = os.path.splitext(os.path.basename(img_path))[0]
    cache_file = os.path.join("results", f"{output_base}_cache.json")

    try:
        if args.from_cache:
            if not os.path.exists(cache_file):
                print(f"Cache not found: {cache_file}. Skipping.")
                continue

            dot_candidates, img, egg_mask, cache_data = load_from_cache(cache_file, img_path)

            if dot_candidates is None:
                print("Could not read the cache. Skipping.")
                continue

            if manual_adjust:
                dot_candidates, keep_current = manual_adjust_dot_candidates(
                    dot_candidates,
                    img,
                    window_name=f"Adjust: {os.path.basename(img_path)}"
                )

                if not keep_current:
                    print("User skipped the current image.")
                    continue

            dot_candidates, img, egg_mask = do_color_sampling(
                dot_candidates,
                img,
                egg_mask,
                erode_ratio=args.erode_ratio
            )

            save_results(
                dot_candidates,
                img,
                egg_mask,
                img_path,
                "results",
                output_base
            )

        else:
            detect_dots_sam(
                img_path,
                erode_ratio=args.erode_ratio,
                manual_adjust=manual_adjust
            )

    except TypeError as e:
        print(f"Processing failed: {e}")
        print("\nAn unexpected 'manual_adjust' keyword usually means detect_dots_sam_v4_lxy.py is out of date.")
        print("Check that detect_dots_sam() accepts manual_adjust and adjusts points between Step 6 and Step 7.")

    except Exception as e:
        print(f"Processing failed: {e}")


print("\n" + "=" * 60)
print("Batch processing complete.")
