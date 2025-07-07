import pandas as pd

# === Compensation Strategy Based on Fuzzy Severity ===
def fault_compensation_from_fuzzy(score):
    if score < 0.3:
        fault_type = "No Fault"
        actions = []
    elif score < 0.7:
        fault_type = "Moderate Fault"
        actions = [
            "Activate Mild Harmonic Filtering",
            "Use ANN-based Monitoring"
        ]
    else:
        fault_type = "Severe Fault"
        actions = [
            "Apply DVR (Dynamic Voltage Restorer)",
            "Apply UPQC (Unified Power Quality Conditioner)",
            "Engage STATCOM",
            "Activate APFs",
            "Use AI Controllers"
        ]
    return fault_type, actions

# === Load CSV ===
csv_path = ''
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print("✅ Columns in input:", df.columns.tolist())

# === Apply Strategy ===
results = []

print("\n📋 Fault Compensation Report Based on Fuzzy Severity:\n")

for idx, row in df.iterrows():
    try:
        true_label = int(row['True_Label'])
        pred_label = int(row['Predicted_Label'])
        prob = float(row['CNN_Probability'])
        fuzzy_score = float(row['Fuzzy_Severity_Score'])
        fuzzy_label = row['Fuzzy_Severity_Label']
    except KeyError as e:
        print(f"❌ Missing column: {e}")
        break

    fault_type, actions = fault_compensation_from_fuzzy(fuzzy_score)

    print(f"--- Sample {idx+1} ---")
    print(f"True Label          : {true_label}")
    print(f"Predicted Label     : {pred_label}")
    print(f"CNN Probability     : {prob:.2f}")
    print(f"Fuzzy Score         : {fuzzy_score:.2f}")
    print(f"⚠️  Detected Fault  : {fault_type}")
    if actions:
        for act in actions:
            print(f"✅ {act}")
    else:
        print("✅ No corrective action needed.")
    print()

    results.append({
        "True_Label": true_label,
        "Predicted_Label": pred_label,
        "CNN_Probability": prob,
        "Fuzzy_Severity_Score": fuzzy_score,
        "Fuzzy_Severity_Label": fuzzy_label,
        "Detected_Fault": fault_type,
        "Corrective_Actions": "; ".join(actions) if actions else "None"
    })

# === Save Output CSV ===
output_df = pd.DataFrame(results)
output_path = ''
output_df.to_csv(output_path, index=False)

print(f"✅ Full fault compensation report saved to:\n{output_path}")
