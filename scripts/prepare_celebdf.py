# scripts/prepare_celebdf.py
import argparse, cv2, re
from pathlib import Path
from tqdm import tqdm

VID_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def infer_label(p: Path) -> str:
    s = f"{p.parent.as_posix()}_{p.stem}".lower()
    # REAL anahtarları
    if any(k in s for k in ["original", "authentic", "real", "youtube"]):
        return "real"
    # FAKE anahtarları
    if any(k in s for k in ["manipulated", "fake", "deepfake", "df", "faceswap", "face2face"]):
        return "fake"
    # Klasör adı real/fake ise onu kullan
    if p.parent.name.lower() == "real":
        return "real"
    if p.parent.name.lower() == "fake":
        return "fake"
    # Bulunamazsa güvenli varsayılan
    return "fake"

def extract_frames(video: Path, out_dir: Path, target_fps: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(native_fps / max(0.1, target_fps))))
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{video.stem}_{idx:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

def main(src: str, out: str, fps: float):
    src_p, out_p = Path(src), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    total, real_cnt, fake_cnt = 0, 0, 0

    for v in tqdm(sorted(src_p.rglob("*")), desc="scan"):
        if v.suffix.lower() in VID_EXT:
            lbl = infer_label(v)
            n = extract_frames(v, out_p / lbl, fps)
            total += n
            if lbl == "real": real_cnt += n
            else:             fake_cnt += n

    print(f"[OK] Kaydedilen kare: {total}")
    print(f"[INFO] Çıktı: {out_p/'real'}  ve  {out_p/'fake'}")
    print(f"[COUNT] real={real_cnt}  fake={fake_cnt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Kök klasör (içinde videolar)")
    ap.add_argument("--out", default="datasets/celebdf_frames")
    ap.add_argument("--fps", type=float, default=2.0)
    a = ap.parse_args()
    main(a.src, a.out, a.fps)
