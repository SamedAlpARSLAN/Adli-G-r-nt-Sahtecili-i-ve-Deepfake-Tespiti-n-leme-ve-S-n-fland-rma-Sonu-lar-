from pathlib import Path
import pandas as pd

res_dir = Path("results")
dfs = []
for csv in sorted(res_dir.glob("metrics.csv")):
    try: dfs.append(pd.read_csv(csv))
    except: pass
if not dfs:
    raise SystemExit("[ERR] results/ altında metrics.csv yok.")
df = pd.concat(dfs, ignore_index=True)
df.sort_values(["dataset","preprocess","time"], inplace=True)
(df).to_csv(res_dir/"metrics_all.csv", index=False)
last = df.groupby(["dataset","preprocess"]).tail(1)
last.to_csv(res_dir/"metrics_last.csv", index=False)
pivot = last.pivot(index="dataset", columns="preprocess", values="accuracy")
pivot.to_csv(res_dir/"metrics_pivot.csv")
print("[OK] metrics_all.csv | metrics_last.csv | metrics_pivot.csv üretildi.")
