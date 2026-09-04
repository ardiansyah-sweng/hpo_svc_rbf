import pandas as pd, os
for mode in ["baseline","isolate_F","isolate_S","isolate_O"]:
    p = f"results/{mode}/summary.csv"
    if not os.path.exists(p):
        print(f"{mode}: FILE HILANG!"); continue
    df = pd.read_csv(p)
    opts = sorted(df['optimizer'].unique())
    print(f"{mode}: {len(df)} baris, {df['optimizer'].nunique()} optimizer, "
          f"{df['dataset'].nunique()} dataset, "
          f"NaN acc: {df['test_acc'].isna().sum()}")
    print(f"    optimizers: {opts}")
    print(f"    cuckoo rows: {(df['optimizer']=='cuckoo').sum()}, sh rows: {(df['optimizer']=='sh').sum()}")