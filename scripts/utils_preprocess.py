import cv2, numpy as np

def gray_world(bgr: np.ndarray) -> np.ndarray:
    b, g, r = [c.astype(np.float32)+1e-6 for c in cv2.split(bgr)]
    m = (b.mean()+g.mean()+r.mean())/3.0
    b *= (m/b.mean()); g *= (m/g.mean()); r *= (m/r.mean())
    out = cv2.merge([b,g,r])
    return np.clip(out,0,255).astype(np.uint8)

def apply_pipeline(bgr: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    out = bgr
    if "grayworld" in mode: out = gray_world(out)
    if "median"    in mode: out = cv2.medianBlur(out, 3)
    if "gaussian"  in mode: out = cv2.GaussianBlur(out, (3,3), 0)
    return out
