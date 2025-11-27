import argparse, shutil, re
from pathlib import Path

IMG_EXT = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

# Ad/klasörden etiket çıkar: _F* -> forged, _O*|orig -> original
PAT_FORGED = re.compile(r'(^|[_-])F([._-]|$)', re.I)
PAT_ORIG   = re.compile(r'(^|[_-])O([._-]|$)|orig', re.I)

def label_from(p: Path):
    s = f"{p.parent.name}_{p.stem}".lower()
    if PAT_FORGED.search(s): return "forged"
    if PAT_ORIG.search(s):   return "original"
    # varyantlar (BC, CR, NA vb.) genellikle sahte
    if re.search(r'\b(bc|cr|na)\d*', s): return "forged"
    return "original"

def main(src, out):
    src, out = Path(src), Path(out)
    if not src.exists():
        raise SystemExit(f"[ERR] Kaynak yok: {src}")
    (out/"original").mkdir(parents=True, exist_ok=True)
    (out/"forged").mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            lbl = label_from(p)
            dst = out/lbl/f"{p.stem}{p.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(p, dst); moved += 1
    print(f"[OK] Kopyalanan görsel: {moved}")
    print(f"[INFO] Çıktı: {out/'original'}  ve  {out/'forged'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CoMoFoD kök klasörü (içinde PNG/JPG olmalı)")
    ap.add_argument("--out", default="datasets/comofod")
    a = ap.parse_args()
    main(a.src, a.out)
