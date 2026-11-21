import os
import glob
import shutil

def cap_val_list(val_lines, target_count=200):
    if len(val_lines) <= target_count:
        return val_lines
    step = len(val_lines) // target_count
    return val_lines[::step][:target_count]

def build_large_area():
    print("=" * 60)
    print("Building Large-Area Specialist Master Dataset...")
    out_dir = os.path.join("dataset", "specialist_large_area")
    os.makedirs(out_dir, exist_ok=True)

    train_lines = []
    val_lines = []

    # 1. Mall Dataset
    mall_train = os.path.join("dataset", "mall_dataset", "train_list.txt")
    mall_val = os.path.join("dataset", "mall_dataset", "val_list.txt")

    if os.path.exists(mall_train):
        with open(mall_train, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    train_lines.append(f"../mall_dataset/{parts[0]} ../mall_dataset/{parts[1]}")

    if os.path.exists(mall_val):
        with open(mall_val, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    val_lines.append(f"../mall_dataset/{parts[0]} ../mall_dataset/{parts[1]}")

    # 2. ShanghaiTech Part B
    shhb_train = os.path.join("dataset", "ShanghaiTech", "part_B", "train_list.txt")
    shhb_val = os.path.join("dataset", "ShanghaiTech", "part_B", "val_list.txt")

    if os.path.exists(shhb_train):
        with open(shhb_train, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    train_lines.append(f"../ShanghaiTech/{parts[0]} ../ShanghaiTech/{parts[1]}")

    if os.path.exists(shhb_val):
        with open(shhb_val, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    val_lines.append(f"../ShanghaiTech/{parts[0]} ../ShanghaiTech/{parts[1]}")

    # 3. JHU-Crowd++ Medium Surveillance subset (16 to 400 count)
    for split_name, lines_list in [("train", train_lines), ("valid", val_lines)]:
        gt_dir = os.path.join("dataset", "Jhu-Crowd++", split_name, "gt_txt")
        if os.path.isdir(gt_dir):
            for gt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
                base = os.path.splitext(os.path.basename(gt_file))[0]
                img_name = base[3:] + ".jpg"
                img_rel = f"../Jhu-Crowd++/{split_name}/images/{img_name}"
                txt_rel = f"../Jhu-Crowd++/{split_name}/gt_txt/{os.path.basename(gt_file)}"
                with open(gt_file, "r") as f:
                    cnt = len(f.readlines())
                if 16 <= cnt <= 400:
                    lines_list.append(f"{img_rel} {txt_rel}")

    capped_val = cap_val_list(val_lines, target_count=200)

    with open(os.path.join(out_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(out_dir, "val_list.txt"), "w") as f:
        f.write("\n".join(capped_val) + "\n")

    print(f"Large-Area Specialist: {len(train_lines)} train, {len(capped_val)} val.")


def build_confined():
    print("=" * 60)
    print("Building Confined-Space Specialist Master Dataset...")
    out_dir = os.path.join("dataset", "specialist_confined")
    os.makedirs(out_dir, exist_ok=True)

    train_lines = []
    val_lines = []

    # 1. SCUT-HEAD dataset
    scut_train = os.path.join("dataset", "scut_head", "train_list.txt")
    scut_val = os.path.join("dataset", "scut_head", "val_list.txt")

    if os.path.exists(scut_train):
        with open(scut_train, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    train_lines.append(f"../scut_head/{parts[0]} ../scut_head/{parts[1]}")

    if os.path.exists(scut_val):
        with open(scut_val, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    val_lines.append(f"../scut_head/{parts[0]} ../scut_head/{parts[1]}")

    # 2. JHU-Crowd++ Sparse / Confined subset (1 to 15 people)
    for split_name, lines_list in [("train", train_lines), ("valid", val_lines)]:
        gt_dir = os.path.join("dataset", "Jhu-Crowd++", split_name, "gt_txt")
        if os.path.isdir(gt_dir):
            for gt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
                base = os.path.splitext(os.path.basename(gt_file))[0]
                img_name = base[3:] + ".jpg"
                img_rel = f"../Jhu-Crowd++/{split_name}/images/{img_name}"
                txt_rel = f"../Jhu-Crowd++/{split_name}/gt_txt/{os.path.basename(gt_file)}"
                with open(gt_file, "r") as f:
                    cnt = len(f.readlines())
                if 1 <= cnt <= 15:
                    lines_list.append(f"{img_rel} {txt_rel}")

    capped_val = cap_val_list(val_lines, target_count=200)

    with open(os.path.join(out_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(out_dir, "val_list.txt"), "w") as f:
        f.write("\n".join(capped_val) + "\n")

    print(f"Confined-Space Specialist: {len(train_lines)} train, {len(capped_val)} val.")


def build_supercrowd():
    print("=" * 60)
    print("Building Supercrowd Specialist Master Dataset...")
    out_dir = os.path.join("dataset", "specialist_supercrowd")
    os.makedirs(out_dir, exist_ok=True)

    train_lines = []
    val_lines = []

    # 1. ShanghaiTech Part A (Extreme Density)
    shha_train = os.path.join("dataset", "ShanghaiTech", "part_A", "train_list.txt")
    shha_val = os.path.join("dataset", "ShanghaiTech", "part_A", "val_list.txt")

    if os.path.exists(shha_train):
        with open(shha_train, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    train_lines.append(f"../ShanghaiTech/{parts[0]} ../ShanghaiTech/{parts[1]}")

    if os.path.exists(shha_val):
        with open(shha_val, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    val_lines.append(f"../ShanghaiTech/{parts[0]} ../ShanghaiTech/{parts[1]}")

    # 2. JHU-Crowd++ Supercrowd subset (>500 heads)
    for split_name, lines_list in [("train", train_lines), ("valid", val_lines)]:
        gt_dir = os.path.join("dataset", "Jhu-Crowd++", split_name, "gt_txt")
        if os.path.isdir(gt_dir):
            for gt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
                base = os.path.splitext(os.path.basename(gt_file))[0]
                img_name = base[3:] + ".jpg"
                img_rel = f"../Jhu-Crowd++/{split_name}/images/{img_name}"
                txt_rel = f"../Jhu-Crowd++/{split_name}/gt_txt/{os.path.basename(gt_file)}"
                with open(gt_file, "r") as f:
                    cnt = len(f.readlines())
                if cnt >= 500:
                    lines_list.append(f"{img_rel} {txt_rel}")

    # 3. NWPU-Crowd
    nwpu_train = os.path.join("dataset", "nwpu_crowd", "train_list.txt")
    nwpu_val = os.path.join("dataset", "nwpu_crowd", "val_list.txt")
    if os.path.exists(nwpu_train):
        with open(nwpu_train, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    train_lines.append(f"../nwpu_crowd/{parts[0]} ../nwpu_crowd/{parts[1]}")
    if os.path.exists(nwpu_val):
        with open(nwpu_val, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    val_lines.append(f"../nwpu_crowd/{parts[0]} ../nwpu_crowd/{parts[1]}")

    capped_val = cap_val_list(val_lines, target_count=200)

    with open(os.path.join(out_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(out_dir, "val_list.txt"), "w") as f:
        f.write("\n".join(capped_val) + "\n")

    print(f"Supercrowd Specialist: {len(train_lines)} train, {len(capped_val)} val.")


if __name__ == "__main__":
    build_large_area()
    build_confined()
    build_supercrowd()
