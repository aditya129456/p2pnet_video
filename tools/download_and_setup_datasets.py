import os
import sys
import time
import zipfile
import urllib.request
import shutil
import glob
import xml.etree.ElementTree as ET
from PIL import Image

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    full = f"{timestamp} {msg}"
    print(full, flush=True)

def download_file(url, out_path):
    log(f"Fetching: {url} -> {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    t0 = time.time()
    last_log_time = t0
    
    with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as out_f:
        total_size = int(resp.headers.get('Content-Length', 0))
        block_size = 2 * 1024 * 1024 # 2MB chunks
        downloaded = 0
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log_time >= 5.0: # Log every 5 seconds
                percent = (downloaded * 100.0 / total_size) if total_size > 0 else 0
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                speed = mb / (now - t0 + 1e-5)
                log(f"Progress: {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%) @ {speed:.2f} MB/s")
                last_log_time = now

    elapsed = time.time() - t0
    final_mb = downloaded / (1024 * 1024)
    log(f"Download complete: {final_mb:.1f} MB in {elapsed:.1f}s ({final_mb/elapsed:.2f} MB/s)")

def extract_zip(zip_path, extract_dir):
    log(f"Extracting {zip_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    log(f"Extracted successfully: {zip_path}")

def setup_scut_head():
    log("=" * 60)
    log("Setting up SCUT-HEAD for Confined-Space Specialist...")
    scut_root = os.path.join("dataset", "scut_head")
    zip_path = os.path.join("dataset", "downloads", "scut_head_part001.zip")
    
    if not os.path.exists(os.path.join(scut_root, "train_list.txt")):
        if not os.path.isdir(scut_root) or len(os.listdir(scut_root)) == 0:
            url = "https://huggingface.co/datasets/hieupth/scut_head/resolve/main/scut_head_part001.zip"
            download_file(url, zip_path)
            extract_zip(zip_path, scut_root)
            if os.path.exists(zip_path):
                os.remove(zip_path)
                log(f"Cleaned up temporary archive: {zip_path}")

        # Locate images and annotations
        all_imgs = glob.glob(os.path.join(scut_root, "**", "*.jpg"), recursive=True)
        all_xmls = glob.glob(os.path.join(scut_root, "**", "*.xml"), recursive=True)
        log(f"Found {len(all_imgs)} images and {len(all_xmls)} XML annotations in SCUT-HEAD.")

        gt_txt_dir = os.path.join(scut_root, "gt_txt")
        os.makedirs(gt_txt_dir, exist_ok=True)

        xml_map = {os.path.splitext(os.path.basename(x))[0]: x for x in all_xmls}
        train_lines = []
        val_lines = []

        valid_count = 0
        for idx, img_path in enumerate(all_imgs):
            base = os.path.splitext(os.path.basename(img_path))[0]
            if base not in xml_map:
                continue
            xml_file = xml_map[base]

            points = []
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        xmin = float(bndbox.find('xmin').text)
                        ymin = float(bndbox.find('ymin').text)
                        xmax = float(bndbox.find('xmax').text)
                        ymax = float(bndbox.find('ymax').text)
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                        points.append((cx, cy))
            except Exception:
                continue

            txt_out_path = os.path.join(gt_txt_dir, f"GT_{base}.txt")
            with open(txt_out_path, "w") as tf:
                for px, py in points:
                    tf.write(f"{px:.2f} {py:.2f}\n")

            rel_img = os.path.relpath(img_path, scut_root).replace("\\", "/")
            rel_txt = os.path.relpath(txt_out_path, scut_root).replace("\\", "/")
            entry = f"{rel_img} {rel_txt}"

            if idx % 5 == 0:
                val_lines.append(entry)
            else:
                train_lines.append(entry)
            valid_count += 1

        with open(os.path.join(scut_root, "train_list.txt"), "w") as f:
            f.write("\n".join(train_lines) + "\n")
        with open(os.path.join(scut_root, "val_list.txt"), "w") as f:
            f.write("\n".join(val_lines) + "\n")

        log(f"[SCUT-HEAD] Processed successfully: {len(train_lines)} train, {len(val_lines)} val.")
    else:
        log("[SCUT-HEAD] Already prepared!")

def setup_nwpu_part1():
    log("=" * 60)
    log("Setting up NWPU-Crowd for Supercrowd Specialist...")
    nwpu_root = os.path.join("dataset", "nwpu_crowd")
    downloads_dir = os.path.join("dataset", "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(nwpu_root, exist_ok=True)

    if not os.path.exists(os.path.join(nwpu_root, "train_list.txt")):
        for fname in ['mats.zip', 'train.txt', 'val.txt']:
            dest = os.path.join(nwpu_root, fname) if not fname.endswith('.zip') else os.path.join(downloads_dir, fname)
            if not os.path.exists(dest) and not os.path.isdir(os.path.join(nwpu_root, "mats")):
                url = f"https://huggingface.co/datasets/Junyu3/NWPU-Crowd/resolve/main/{fname}"
                download_file(url, dest)
                if fname.endswith('.zip'):
                    extract_zip(dest, nwpu_root)
                    if os.path.exists(dest):
                        os.remove(dest)

        img_zip = os.path.join(downloads_dir, "images_part1.zip")
        images_dir = os.path.join(nwpu_root, "images")
        if not os.path.isdir(images_dir) or len(glob.glob(os.path.join(images_dir, "*.jpg"))) < 500:
            url = "https://huggingface.co/datasets/Junyu3/NWPU-Crowd/resolve/main/images_part1.zip"
            download_file(url, img_zip)
            extract_zip(img_zip, nwpu_root)
            if os.path.exists(img_zip):
                os.remove(img_zip)
                log(f"Cleaned up temporary archive: {img_zip}")

        from scipy.io import loadmat
        mats_dir = os.path.join(nwpu_root, "mats")
        gt_txt_dir = os.path.join(nwpu_root, "gt_txt")
        os.makedirs(gt_txt_dir, exist_ok=True)

        available_imgs = glob.glob(os.path.join(images_dir, "*.jpg"))
        log(f"NWPU-Crowd images extracted: {len(available_imgs)}")

        train_lines = []
        val_lines = []

        for idx, img_path in enumerate(available_imgs):
            base = os.path.splitext(os.path.basename(img_path))[0]
            mat_file = os.path.join(mats_dir, f"{base}.mat")
            if not os.path.exists(mat_file):
                continue

            try:
                mat_data = loadmat(mat_file)
                coords = mat_data.get('annPoints', [])
            except Exception:
                continue

            txt_out = os.path.join(gt_txt_dir, f"GT_{base}.txt")
            with open(txt_out, "w") as tf:
                for pt in coords:
                    tf.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")

            rel_img = f"images/{os.path.basename(img_path)}"
            rel_txt = f"gt_txt/GT_{base}.txt"
            entry = f"{rel_img} {rel_txt}"

            if idx % 5 == 0:
                val_lines.append(entry)
            else:
                train_lines.append(entry)

        with open(os.path.join(nwpu_root, "train_list.txt"), "w") as f:
            f.write("\n".join(train_lines) + "\n")
        with open(os.path.join(nwpu_root, "val_list.txt"), "w") as f:
            f.write("\n".join(val_lines) + "\n")

        log(f"[NWPU-Crowd] Processed successfully: {len(train_lines)} train, {len(val_lines)} val.")
    else:
        log("[NWPU-Crowd] Already prepared!")

if __name__ == "__main__":
    setup_scut_head()
    setup_nwpu_part1()
    log("=" * 60)
    log("All external datasets downloaded and prepared successfully!")
