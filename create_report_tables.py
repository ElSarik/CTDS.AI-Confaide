import os
import json
from glob import glob
import pandas as pd
from scipy.stats import pearsonr

# Path to eval_results directory
EVAL_RESULTS_DIR = 'eval_results'

# Tiers to look for
TIERS = ['1', '2a', '2b']
# Manual display names for tiers
TIER_DISPLAY_NAMES = {
    '1': 'Tier 1: Info-Sensitivity',
    '2a': 'Tier 2a: InfoFlow-Expectation',
    '2b': 'Tier 2b: InfoFlow-Expectation',
}

# Store means per model and tier
model_means = {}
model_names = []

# Normalization function for tier 1
normalize_t1 = lambda rating: (2.5 - rating) / 1.5 * 100.0

# --- Read human results ---
human_means = {}
# Tier 1
try:
    with open('benchmark/tier_1_labels.txt', 'r') as f:
        t1_labels = [float(line.strip()) for line in f if line.strip()]
    t1_labels_norm = [normalize_t1(v) for v in t1_labels]
    human_means['1'] = sum(t1_labels_norm) / len(t1_labels_norm) if t1_labels_norm else None
except Exception as e:
    print(f'Error reading tier_1_labels.txt: {e}')
    human_means['1'] = None
# Tier 2a and 2b
try:
    with open('benchmark/tier_2_labels.txt', 'r') as f:
        t2_labels = [float(line.strip()) for line in f if line.strip()]
    t2_mean = sum(t2_labels) / len(t2_labels) if t2_labels else None
    human_means['2a'] = t2_mean
    human_means['2b'] = t2_mean
except Exception as e:
    print(f'Error reading tier_2_labels.txt: {e}')
    human_means['2a'] = None
    human_means['2b'] = None

# --- For Table 1: Store all model and human values for correlation ---
# {model: {tier: ([model_values], [human_values])}}
model_human_values = {}

# Only use tiers 1, 2a, 2b for Table 1
TIERS_CORR = ['1', '2a', '2b']

# Read human values for all tiers needed for correlation
human_values_corr = {}
# Tier 1
try:
    with open('benchmark/tier_1_labels.txt', 'r') as f:
        human_values_corr['1'] = [float(line.strip()) for line in f if line.strip()]
except Exception as e:
    print(f'Error reading tier_1_labels.txt for correlation: {e}')
    human_values_corr['1'] = []
# Tier 2a and 2b (identical)
try:
    with open('benchmark/tier_2_labels.txt', 'r') as f:
        t2_vals = [float(line.strip()) for line in f if line.strip()]
    human_values_corr['2a'] = t2_vals
    human_values_corr['2b'] = t2_vals
except Exception as e:
    print(f'Error reading tier_2_labels.txt for correlation: {e}')
    human_values_corr['2a'] = []
    human_values_corr['2b'] = []

# Iterate over all model directories
for model_dir in os.listdir(EVAL_RESULTS_DIR):
    model_path = os.path.join(EVAL_RESULTS_DIR, model_dir)
    if not os.path.isdir(model_path):
        continue
    model_names.append(model_dir)
    model_means[model_dir] = {}
    model_human_values[model_dir] = {}
    for tier in TIERS:
        tier_dir = os.path.join(model_path, f'Tier {tier}')
        if not os.path.isdir(tier_dir):
            continue
        # Find all final_report json files in this tier directory
        for report_file in glob(os.path.join(tier_dir, 'final_report_*.json')):
            try:
                with open(report_file, 'r') as f:
                    data = json.load(f)
                # Some files may be a list, some may be a dict
                if isinstance(data, list):
                    model_values = [entry.get('model') for entry in data if isinstance(entry, dict) and isinstance(entry.get('model'), (int, float))]
                elif isinstance(data, dict):
                    model_values = [v.get('model') for v in data.values() if isinstance(v, dict) and isinstance(v.get('model'), (int, float))]
                else:
                    model_values = []
                # For means (Table 2)
                if model_values and tier in TIERS:
                    if tier == '1':
                        # Normalize for tier 1
                        model_values_norm = [normalize_t1(v) for v in model_values]
                        mean_value = sum(model_values_norm) / len(model_values_norm)
                    else:
                        mean_value = sum(model_values) / len(model_values)
                    model_means[model_dir][tier] = mean_value
                # For correlation (Table 1)
                if model_values and tier in TIERS_CORR:
                    # Align with human values (truncate to shortest)
                    hvals = human_values_corr.get(tier, [])
                    n = min(len(model_values), len(hvals))
                    if n > 1:
                        model_human_values[model_dir][tier] = (model_values[:n], hvals[:n])
            except Exception as e:
                print(f'Error reading {report_file}: {e}')

# Remove models with no data at all
model_names = [m for m in model_names if model_means[m]]
model_human_values = {m: v for m, v in model_human_values.items() if m in model_names}

# Table 1: Pearson correlation table
corr_rows = []
corr_tiers = ['1', '2a', '2b']
corr_header = ['Tier'] + model_names
for tier in corr_tiers:
    row = [TIER_DISPLAY_NAMES.get(tier, f'Tier {tier}') if tier in TIER_DISPLAY_NAMES else f'Tier {tier}']
    for model in model_names:
        vals = model_human_values.get(model, {}).get(tier)
        if vals and len(vals[0]) > 1 and len(vals[1]) > 1:
            try:
                corr, _ = pearsonr(vals[0], vals[1])
            except Exception:
                corr = float('nan')
            row.append(corr)
        else:
            row.append(float('nan'))
    corr_rows.append(row)

try:
    df_corr = pd.DataFrame(corr_rows, columns=corr_header)
    df_corr.to_excel('Table_1_Pearson_Correlation.xlsx', index=False)
    print('Table 1 results table saved to Table_1_Pearson_Correlation.xlsx')
    print('Table 1: Pearson Correlation Table')
    print(df_corr)

    print(" ")
    print("--------------------------------")
    print(" ")
except Exception as e:
    print(f'Error creating correlation table: {e}')

# Table 2: Means table
rows = []
for tier in TIERS:
    excel_row = [TIER_DISPLAY_NAMES.get(tier, tier)]
    hmean = human_means.get(tier)
    excel_row.append(hmean if hmean is not None else None)
    for model in model_names:
        mean = model_means[model].get(tier)
        excel_row.append(mean if mean is not None else None)
    rows.append(excel_row)

excel_header = ['Tier', 'Human'] + model_names
try:
    df = pd.DataFrame(rows, columns=excel_header)
    df.to_excel('Table_2_InfoFlow_Expectation.xlsx', index=False)
    print('Table 2 results table saved to Table_2_InfoFlow_Expectation.xlsx')
    print('Table 2: InfoFlow Expectation Table')
    print(df)

    print(" ")
    print("--------------------------------")
    print(" ")
except Exception as e:
    print(f'Error saving Excel file: {e}')

# Table 3: Direct metric extraction from JSON files
# Metric definitions from notes.txt
metric_specs = [
    # (Row label, JSON filename part, JSON key)
    ("Leakage thru. String Match", "free-response_metrics_string-match", "free-response-string-match_mean"),
    ("Leakage thru. String Match (Worst case)", "free-response_metrics_string-match", "free-response-string-match_worst_case"),
    ("ToM. Information Access Err.", "info-accessibility_metrics_error", "info-accessibility-error_mean"),
    ("ToM. Information Access Err. (Worst case)", "info-accessibility_metrics_error", "info-accessibility-error_worst_case"),
    ("ToM. Private Info. Access Err.", "privacy-sharing_metrics_error", "privacy-sharing-error_mean"),
    ("ToM. Private Info. Access Err. (Worst case)", "privacy-sharing_metrics_error", "privacy-sharing-error_worst_case"),
]

# Build Table 3 rows
rows3 = []
for row_label, json_part, key in metric_specs:
    row = [row_label]
    for model in model_names:
        # Search for the JSON file in Tier 3 (not Tier 4)
        tier3_dir = os.path.join(EVAL_RESULTS_DIR, model, 'Tier 3')
        if not os.path.isdir(tier3_dir):
            row.append(None)
            continue
        # Find the file containing the json_part and starting with 'final_report'
        found_file = None
        for fname in os.listdir(tier3_dir):
            if fname.startswith('final_report') and json_part in fname and fname.endswith('.json'):
                found_file = os.path.join(tier3_dir, fname)
                break
        if not found_file:
            row.append(None)
            continue
        # Read the value from the file
        try:
            with open(found_file, 'r') as f:
                data = json.load(f)
            value = data.get(key, None)
            row.append(value)
        except Exception as e:
            print(f'Error reading {found_file} for {key}: {e}')
            row.append(None)
    rows3.append(row)

header3 = ["Metric"] + model_names
try:
    df3 = pd.DataFrame(rows3, columns=header3)
    df3.to_excel('Table_3_ToM.xlsx', index=False)
    print('Table 3 results table saved to Table_3_ToM.xlsx')
    print('Table 3: Theory of Mind Extraction Table')
    print(df3)

    print(" ")
    print("--------------------------------")
    print(" ")
except Exception as e:
    print(f'Error creating Table 3: {e}')

# Table 4: Direct metric extraction from JSON files for Tier 4
metric_specs_4 = [
    ("Act. Item Leaks Secret", "action-item_metrics_has_private_info", "action-item-has_private_info_mean"),
    ("Act. Item Leaks Secret (Worst case)", "action-item_metrics_has_private_info", "action-item-has_private_info_worst_case"),
    ("Act. Item Omits Public Information", "action-item_metrics_no_public_info", "action-item-no_public_info_mean"),
    ("Act. Item Omits Public Information (Worst case)", "action-item_metrics_no_public_info", "action-item-no_public_info_worst_case"),
    ("Act. Item Leaks Secret or Omits Info.", "action-item_metrics_error", "action-item-error_mean"),
    ("Act. Item Leaks Secret or Omits Info. (Worst case)", "action-item_metrics_error", "action-item-error_worst_case"),
    ("Summary Leaks Secret", "meeting-summary_metrics_has_private_info", "meeting-summary-has_private_info_mean"),
    ("Summary Leaks Secret (Worst case)", "meeting-summary_metrics_has_private_info", "meeting-summary-has_private_info_worst_case"),
    ("Summary Omits Public Information", "meeting-summary_metrics_no_public_info", "meeting-summary-no_public_info_mean"),
    ("Summary Omits Public Information (Worst case)", "meeting-summary_metrics_no_public_info", "meeting-summary-no_public_info_worst_case"),
    ("Summary Leaks Secret or Omits Info.", "meeting-summary_metrics_error", "meeting-summary-error_mean"),
    ("Summary Leaks Secret or Omits Info. (Worst case)", "meeting-summary_metrics_error", "meeting-summary-error_worst_case"),
]

rows4 = []
for row_label, json_part, key in metric_specs_4:
    row = [row_label]
    for model in model_names:
        tier4_dir = os.path.join(EVAL_RESULTS_DIR, model, 'Tier 4')
        if not os.path.isdir(tier4_dir):
            row.append(None)
            continue
        found_file = None
        for fname in os.listdir(tier4_dir):
            if fname.startswith('final_report') and json_part in fname and fname.endswith('.json'):
                found_file = os.path.join(tier4_dir, fname)
                break
        if not found_file:
            row.append(None)
            continue
        try:
            with open(found_file, 'r') as f:
                data = json.load(f)
            value = data.get(key, None)
            row.append(value)
        except Exception as e:
            print(f'Error reading {found_file} for {key}: {e}')
            row.append(None)
    rows4.append(row)

header4 = ["Metric"] + model_names
try:
    df4 = pd.DataFrame(rows4, columns=header4)
    df4.to_excel('Table_4_Action_Summary.xlsx', index=False)
    print('Table 4 results table saved to Table_4_Action_Summary.xlsx')
    print('Table 4: Action/Summary Extraction Table')
    print(df4)
    print(" ")
    print("--------------------------------")
    print(" ")
except Exception as e:
    print(f'Error creating Table 4: {e}')