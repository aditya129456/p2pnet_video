import os
import sys
import time
import shutil
import subprocess
import torch

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    full = f"{timestamp} {msg}"
    print(full, flush=True)

def run_training_campaign(campaign_name, data_root, output_dir, final_weight_path, epochs, lr, lr_backbone, eos_coef=0.5, point_loss_coef=0.0002, eval_freq=3):
    log("=" * 70)
    log(f"STARTING CAMPAIGN: {campaign_name}")
    log(f"Data: {data_root} | Epochs: {epochs} | LR: {lr} (Backbone: {lr_backbone}) | EOS: {eos_coef}")
    log("=" * 70)

    if os.path.exists(final_weight_path):
        size_mb = os.path.getsize(final_weight_path) / (1024 * 1024)
        log(f"Checkpoint {final_weight_path} already exists ({size_mb:.1f} MB). Skipping training.")
        return

    ckpt_dir = os.path.join(output_dir, "checkpoint")
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        "--data_root", data_root,
        "--dataset_file", "CUSTOM",
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--lr_backbone", str(lr_backbone),
        "--eos_coef", str(eos_coef),
        "--point_loss_coef", str(point_loss_coef),
        "--batch_size", "8",
        "--eval_freq", str(eval_freq),
        "--num_workers", "0",
        "--frozen_weights", "./weights/best_mae.pth",
        "--output_dir", output_dir,
        "--checkpoints_dir", ckpt_dir,
        "--tensorboard_dir", runs_dir,
    ]

    log(f"Executing: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode != 0:
        log(f"ERROR: {campaign_name} failed with returncode {res.returncode}")
        sys.exit(res.returncode)

    best_ckpt = os.path.join(ckpt_dir, "best_mae.pth")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, "latest.pth")

    if os.path.exists(best_ckpt):
        os.makedirs(os.path.dirname(final_weight_path), exist_ok=True)
        # Strip optimizer states to keep weights clean (~82-86 MB)
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        clean_model = ckpt_data["model"] if "model" in ckpt_data else ckpt_data
        torch.save({"model": clean_model}, final_weight_path)

        size_mb = os.path.getsize(final_weight_path) / (1024 * 1024)
        log(f"SUCCESS: {campaign_name} completed in {elapsed/60:.1f} mins!")
        log(f"Landed clean deployment checkpoint: {final_weight_path} ({size_mb:.1f} MB)")
    else:
        log(f"ERROR: No checkpoint found at {best_ckpt}")
        sys.exit(1)

def verify_all_weights():
    log("=" * 70)
    log("VERIFYING ALL 4 P2PNET WEIGHTS IN ./weights/")
    log("=" * 70)

    from surveillance.detector import CrowdDetector
    import numpy as np

    models_to_test = [
        ("Tier 4: Generalist (Fallback)", "./weights/best_mae.pth"),
        ("Tier 2: Large-Area Specialist", "./weights/p2pnet_large_area.pth"),
        ("Tier 3: Confined-Space Specialist", "./weights/p2pnet_confined.pth"),
        ("Tier 1: Supercrowd Specialist", "./weights/p2pnet_supercrowd.pth"),
    ]

    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    all_passed = True
    for name, weight_path in models_to_test:
        if not os.path.exists(weight_path):
            log(f"[FAIL] {name}: File not found at {weight_path}")
            all_passed = False
            continue

        size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        log(f"Testing {name} ({weight_path}, {size_mb:.1f} MB)...")
        try:
            detector = CrowdDetector(weight_path=weight_path, precision="fp16")
            points, scores, latency = detector.detect(dummy_frame)
            log(f"[PASS] {name} initialized & ran successfully! Latency: {latency:.1f} ms | Detected: {len(points)} points")
        except Exception as e:
            log(f"[FAIL] {name} encountered error: {e}")
            all_passed = False

    return all_passed

def main():
    log("=" * 70)
    log("P2PNET MULTI-SPECIALIST AUTONOMOUS TRAINING ORCHESTRATOR")
    log("=" * 70)

    # 1. Campaign 1: Large-Area Specialist (Skipped if already deployed)
    run_training_campaign(
        campaign_name="Large-Area Specialist",
        data_root="./dataset/specialist_large_area",
        output_dir="./logs/specialist_large_area",
        final_weight_path="./weights/p2pnet_large_area.pth",
        epochs=15,
        lr=0.00001,
        lr_backbone=0.000001,
        eos_coef=0.5,
        point_loss_coef=0.0002,
        eval_freq=3,
    )

    # 2. Campaign 2: Confined-Space Specialist
    run_training_campaign(
        campaign_name="Confined-Space Specialist",
        data_root="./dataset/specialist_confined",
        output_dir="./logs/specialist_confined",
        final_weight_path="./weights/p2pnet_confined.pth",
        epochs=15,
        lr=0.00001,
        lr_backbone=0.000001,
        eos_coef=0.75,
        point_loss_coef=0.0002,
        eval_freq=3,
    )

    # 3. Campaign 3: Supercrowd Specialist
    run_training_campaign(
        campaign_name="Supercrowd Specialist",
        data_root="./dataset/specialist_supercrowd",
        output_dir="./logs/specialist_supercrowd",
        final_weight_path="./weights/p2pnet_supercrowd.pth",
        epochs=15,
        lr=0.00005,
        lr_backbone=0.000005,
        eos_coef=0.35,
        point_loss_coef=0.0003,
        eval_freq=3,
    )

    # 4. Final Verification
    success = verify_all_weights()
    if success:
        log("=" * 70)
        log("ALL 3 P2PNET SPECIALISTS SUCCESSFULLY TRAINED, VERIFIED, AND DEPLOYED!")
        log("Checkpoints in weights/:")
        for f in os.listdir("weights"):
            if f.endswith(".pth"):
                s = os.path.getsize(os.path.join("weights", f)) / (1024 * 1024)
                log(f"  - {f} ({s:.1f} MB)")
        log("=" * 70)
    else:
        log("WARNING: Verification failed on one or more models.")

if __name__ == "__main__":
    main()
