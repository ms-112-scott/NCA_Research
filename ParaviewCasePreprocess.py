import os
import sys
from paraview.simple import *

# 設定案例資料夾
case_root = r"C:\Users\GAI\Desktop\Scott\NCA_rhgh\00_RAW_FloorPLan_CFD_CASE"
macro_path =r"C:\Users\GAI\Desktop\Scott\case2tensor.py"
output_path = r"C:\Users\GAI\Desktop\Scott\NCA_Research\dataset\FP_case_tensor"
# 找出所有 .foam 檔案
foam_cases = []
for root, dirs, files in os.walk(case_root):
    for file in files:
        if file.endswith(".foam"):
            foam_cases.append(os.path.join(root, file))


# === 逐一處理每個 .foam 案例 ===
for i, foam_file in enumerate(foam_cases):

    # 取得檔名 prefix，例如 case_A_123.foam → A_123
    prefix = os.path.splitext(os.path.basename(foam_file))[0]
    prefix = "_".join(prefix.split('_')[1:])
    output_dir = os.path.join(output_path, prefix)

    # === 如果已經有輸出資料夾，就跳過 ===
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"[{i+1}/{len(foam_cases)}] Skipping {prefix}: output already exists.")
        continue

    print(f"[{i+1}/{len(foam_cases)}] Processing {prefix}...")

    try:
        # === 讀取 .foam ===
        reader = OpenFOAMReader(FileName=foam_file)

        # === 執行 macro（會使用 prefix / output_dir）===
        exec(open(macro_path).read())

        # === 清除場景物件 ===
        Delete(reader)
        del readerpkk

    except Exception as e:
        print(f"⚠️  Error processing {prefix}: {e}")
        sys.exit(1)

print("✅ All cases processed.")
