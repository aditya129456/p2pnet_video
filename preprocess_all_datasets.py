import os
import glob
import numpy as np
from PIL import Image
from scipy.io import loadmat

def preprocess_shanghaitech():
    print("=" * 60)
    print("Preprocessing ShanghaiTech Dataset...")
    sh_root = os.path.join("dataset", "ShanghaiTech")
    
    master_train_lines = []
    master_val_lines = []

    for part in ["part_A", "part_B"]:
        part_dir = os.path.join(sh_root, part)
        if not os.path.isdir(part_dir):
            print(f"Skipping {part}: Directory not found.")
            continue

        for split, is_train in [("train_data", True), ("test_data", False)]:
            split_dir = os.path.join(part_dir, split)
            img_dir = os.path.join(split_dir, "images")
            gt_dir = os.path.join(split_dir, "ground-truth")
            out_gt_txt_dir = os.path.join(split_dir, "gt_txt")
            os.makedirs(out_gt_txt_dir, exist_ok=True)

            if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
                print(f"Skipping {part}/{split}: missing images or ground-truth folder.")
                continue

            split_list_lines = []
            mat_files = sorted(glob.glob(os.path.join(gt_dir, "GT_*.mat")))
            print(f"Processing {part} {split}: {len(mat_files)} files...")

            for mat_path in mat_files:
                base_name = os.path.splitext(os.path.basename(mat_path))[0] # e.g. GT_IMG_1
                img_name = base_name[3:] + ".jpg" # IMG_1.jpg
                img_full_path = os.path.join(img_dir, img_name)

                if not os.path.exists(img_full_path):
                    continue

                # Load .mat coordinates
                mat_data = loadmat(mat_path)
                coords = mat_data['image_info'][0][0][0][0][0]

                # Write out as x y text format
                txt_out_path = os.path.join(out_gt_txt_dir, f"{base_name}.txt")
                with open(txt_out_path, "w") as f:
                    for i in range(len(coords)):
                        x, y = coords[i][0], coords[i][1]
                        f.write(f"{x:.2f} {y:.2f}\n")

                # Store relative paths from dataset/ShanghaiTech
                rel_img = f"{part}/{split}/images/{img_name}".replace("\\", "/")
                rel_txt = f"{part}/{split}/gt_txt/{base_name}.txt".replace("\\", "/")
                entry = f"{rel_img} {rel_txt}"
                split_list_lines.append(entry)

                if is_train:
                    master_train_lines.append(entry)
                else:
                    master_val_lines.append(entry)

            # Per-split list file inside part folder
            list_name = "train_list.txt" if is_train else "val_list.txt"
            with open(os.path.join(part_dir, list_name), "w") as f:
                f.write("\n".join(split_list_lines) + "\n")

    # Master list files across Part A and Part B
    with open(os.path.join(sh_root, "train_list.txt"), "w") as f:
        f.write("\n".join(master_train_lines) + "\n")
    with open(os.path.join(sh_root, "val_list.txt"), "w") as f:
        f.write("\n".join(master_val_lines) + "\n")

    print(f"ShanghaiTech finished! Total train: {len(master_train_lines)}, Total val: {len(master_val_lines)}")


def preprocess_mall():
    print("=" * 60)
    print("Preprocessing MALL Dataset...")
    mall_root = os.path.join("dataset", "mall_dataset")
    gt_mat_path = os.path.join(mall_root, "mall_gt.mat")
    frames_dir = os.path.join(mall_root, "frames")
    out_gt_txt_dir = os.path.join(mall_root, "gt_txt")
    os.makedirs(out_gt_txt_dir, exist_ok=True)

    if not os.path.exists(gt_mat_path) or not os.path.isdir(frames_dir):
        print("Skipping MALL: mall_gt.mat or frames/ directory missing.")
        return

    data = loadmat(gt_mat_path)
    frames_gt = data['frame'][0]
    total_frames = len(frames_gt)
    print(f"Processing MALL: {total_frames} frames...")

    train_lines = []
    val_lines = []

    for i in range(total_frames):
        seq_num = i + 1
        img_name = f"seq_{seq_num:06d}.jpg"
        txt_name = f"GT_seq_{seq_num:06d}.txt"
        img_full_path = os.path.join(frames_dir, img_name)
        txt_out_path = os.path.join(out_gt_txt_dir, txt_name)

        if not os.path.exists(img_full_path):
            continue

        locs = frames_gt[i][0][0][0] # (N, 2) array of (x, y) coordinates
        with open(txt_out_path, "w") as f:
            for pt in locs:
                f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")

        rel_img = f"frames/{img_name}".replace("\\", "/")
        rel_txt = f"gt_txt/{txt_name}".replace("\\", "/")
        entry = f"{rel_img} {rel_txt}"

        # Benchmark split: 1-800 train, 801-2000 val
        if seq_num <= 800:
            train_lines.append(entry)
        else:
            val_lines.append(entry)

    with open(os.path.join(mall_root, "train_list.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(mall_root, "val_list.txt"), "w") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"MALL Dataset finished! Train frames: {len(train_lines)}, Val frames: {len(val_lines)}")


def preprocess_jhu():
    print("=" * 60)
    print("Preprocessing JHU-CROWD++ Dataset...")
    jhu_root = os.path.join("dataset", "Jhu-Crowd++")

    for split_src, split_name in [("train", "train"), ("valid", "val")]:
        split_dir = os.path.join(jhu_root, split_src)
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        out_gt_txt_dir = os.path.join(split_dir, "gt_txt")
        os.makedirs(out_gt_txt_dir, exist_ok=True)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            print(f"Skipping JHU {split_src}: missing images or labels folder.")
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"Processing JHU-CROWD++ {split_src}: {len(img_files)} images...")

        list_lines = []
        for img_path in img_files:
            fname = os.path.basename(img_path)
            base_name = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            with Image.open(img_path) as img:
                width, height = img.size

            txt_out_path = os.path.join(out_gt_txt_dir, f"GT_{base_name}.txt")
            coords = []

            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class, x_center, y_center, w, h (normalized)
                            xc, yc = float(parts[1]), float(parts[2])
                            px = xc * width
                            py = yc * height
                            coords.append((px, py))

            with open(txt_out_path, "w") as f:
                for px, py in coords:
                    f.write(f"{px:.2f} {py:.2f}\n")

            rel_img = f"{split_src}/images/{fname}".replace("\\", "/")
            rel_txt = f"{split_src}/gt_txt/GT_{base_name}.txt".replace("\\", "/")
            list_lines.append(f"{rel_img} {rel_txt}")

        out_list_file = "train_list.txt" if split_name == "train" else "val_list.txt"
        with open(os.path.join(jhu_root, out_list_file), "w") as f:
            f.write("\n".join(list_lines) + "\n")

        print(f"JHU-CROWD++ {split_name} finished! Total: {len(list_lines)}")


if __name__ == "__main__":
    preprocess_shanghaitech()
    preprocess_mall()
    preprocess_jhu()
    print("=" * 60)
    print("Phase 1 complete! All datasets preprocessed and standardized successfully.")
