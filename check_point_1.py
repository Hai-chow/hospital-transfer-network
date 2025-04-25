import os
import pandas as pd

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 CSV 文件的完整路径
encounters_path = os.path.join(script_dir, "Hospital_Transfers_by_Major_Diagnostic_Category__MDC__-Encounters.csv")
outcomes_path = os.path.join(script_dir, "Hospital_Transfers_by_Major_Diagnostic_Category__MDC__-Outcomes.csv")
locations_path = os.path.join(script_dir, "Hospital_Transfers_by_Major_Diagnostic_Category__MDC__-Locations.csv")

# 加载数据
df_encounters = pd.read_csv(encounters_path)
df_outcomes = pd.read_csv(outcomes_path)
df_locations = pd.read_csv(locations_path)

# 显示前几行
print("Encounters:")

print(df_encounters.head())

print("\nOutcomes:")
print(df_outcomes.head())

print("\nLocations:")
print(df_locations.head())