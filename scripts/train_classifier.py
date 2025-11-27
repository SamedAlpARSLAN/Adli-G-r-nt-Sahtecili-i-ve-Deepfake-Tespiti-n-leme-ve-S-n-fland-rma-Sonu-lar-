import argparse, time, random, numpy as np, pandas as pd, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import cv2
from utils_preprocess import apply_pipeline

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class Preproc:
    def __init__(self, mode="none"): self.mode = mode
    def __call__(self, img_pil: Image.Image):
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        bgr = apply_pipeline(bgr, self.mode)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

def make_loaders(root, batch_size, preproc, seed=42):
    tfm_list = []
    if preproc!="none": tfm_list.append(Preproc(preproc))
    tfm_list += [transforms.Resize((224,224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    tfm = transforms.Compose(tfm_list)
    ds = datasets.ImageFolder(root=root, transform=tfm)
    y = [ds.samples[i][1] for i in range(len(ds))]
    idx_train, idx_temp = train_test_split(np.arange(len(ds)), test_size=0.30, stratify=y, random_state=seed)
    y_temp = [y[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    tr, va, te = Subset(ds, idx_train), Subset(ds, idx_val), Subset(ds, idx_test)
    return (DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True),
            DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=2),
            DataLoader(te,batch_size=batch_size,shuffle=False,num_workers=2),
            ds.classes)

def train_eval(data_root, dataset_name, preproc, epochs, batch, lr, device, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, classes = make_loaders(data_root, batch, preproc)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_val, best_state = 0.0, None

    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        model.eval(); y_true=[]; y_pred=[]
        with torch.no_grad():
            for x, y in val_loader:
                x=x.to(device); logits = model(x).cpu()
                y_true += list(y.numpy()); y_pred += list(logits.argmax(1).numpy())
        acc = accuracy_score(y_true, y_pred)
        if acc>best_val: best_val, best_state = acc, model.state_dict()
        print(f"Epoch {ep}/{epochs}  val_acc={acc:.4f}")

    model.load_state_dict(best_state)
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x, y in test_loader:
            x=x.to(device); logits = model(x).cpu()
            y_true += list(y.numpy()); y_pred += list(logits.argmax(1).numpy())
    acc = accuracy_score(y_true, y_pred)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    row = {"dataset":dataset_name,"data_root":data_root,"preprocess":preproc,
           "epochs":epochs,"batch":batch,"accuracy":round(float(acc),4),
           "n_test":len(y_true),"time":stamp}
    df = pd.DataFrame([row])
    csv = outdir/"metrics.csv"
    df.to_csv(csv, mode="a", index=False, header=not csv.exists())
    (outdir/f"{dataset_name}_{preproc}_report.txt").write_text(
        classification_report(y_true,y_pred,target_names=classes), encoding="utf-8")
    cm = confusion_matrix(y_true,y_pred)
    np = __import__("numpy"); np.savetxt(outdir/f"{dataset_name}_{preproc}_cm.txt", cm, fmt="%d")
    print("[OK] Test accuracy:", acc)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="ImageFolder kökü (class alt klasörleri)")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--preprocess", default="none", help="none | grayworld+median | grayworld | gaussian | median")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed",   type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_eval(a.data_root, a.dataset_name, a.preprocess, a.epochs, a.batch, a.lr, dev, a.outdir)
