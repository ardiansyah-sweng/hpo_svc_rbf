import pandas as pd, os
for mode in ["baseline","isolate_F","isolate_S","isolate_O"]:
    p = f"results/{mode}/summary.csv"
    if not os.path.exists(p):
        print(f"{mode}: FILE HILANG!")
        continue
    df = pd.read_csv(p)
    print(f"{mode}: {len(df)} baris, "
          f"{df['optimizer'].nunique()} optimizer, "
          f"{df['dataset'].nunique()} dataset, "
          f"seed unik: split={df['seed_split'].nunique()} "
          f"cv={df['seed_cv'].nunique()} opt={df['seed_opt'].nunique()}, "
          f"NaN acc: {df['test_acc'].isna().sum()}")