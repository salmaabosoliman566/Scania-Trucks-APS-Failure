import csv
import json
import os

top_features = ['bj_000', 'aq_000', 'ci_000', 'dn_000', 'ap_000', 'bb_000', 'bv_000', 'bu_000', 'cq_000', 'ck_000', 'bt_000', 'aa_000', 'bh_000', 'ah_000', 'cc_000', 'bg_000', 'an_000', 'bx_000', 'ao_000', 'cs_002']

sensor_meta = {
    'bj_000': {'name': 'Compressor Discharge Pressure Cycle', 'unit': 'cycles', 'category': 'Pneumatic Delivery'},
    'aq_000': {'name': 'Air Tank Delivery Pressure Monitor', 'unit': 'kPa', 'category': 'Pneumatic Delivery'},
    'ci_000': {'name': 'Air Dryer Regeneration Counter', 'unit': 'pulses', 'category': 'Pneumatic Delivery'},
    'dn_000': {'name': 'Brake Actuator Pressure Differential', 'unit': 'mbar', 'category': 'Braking Circuit'},
    'ap_000': {'name': 'Reservoir Main Flow Velocity', 'unit': 'L/min', 'category': 'Pneumatic Delivery'},
    'bb_000': {'name': 'Auxiliary Consumer Flow Supply', 'unit': 'L/min', 'category': 'Braking Circuit'},
    'bv_000': {'name': 'Primary Brake Circuit Pressure', 'unit': 'kPa', 'category': 'Braking Circuit'},
    'bu_000': {'name': 'Secondary Brake Circuit Pressure', 'unit': 'kPa', 'category': 'Braking Circuit'},
    'cq_000': {'name': 'Multi-Circuit Protection Valve State', 'unit': 'ratio', 'category': 'Braking Circuit'},
    'ck_000': {'name': 'Compressor Duty Cycle Accumulator', 'unit': 'sec', 'category': 'Compressor & Governor'},
    'bt_000': {'name': 'Governor Cut-Out Pressure Setpoint', 'unit': 'bar', 'category': 'Compressor & Governor'},
    'aa_000': {'name': 'Master Air Pressure Sensor', 'unit': 'bar', 'category': 'Pneumatic Delivery'},
    'bh_000': {'name': 'Pneumatic Suspension Bellows Pressure', 'unit': 'kPa', 'category': 'Auxiliary & Chassis'},
    'ah_000': {'name': 'Ambient Temp & Compressor Inlet', 'unit': '°C count', 'category': 'Compressor & Governor'},
    'cc_000': {'name': 'Purge Valve Operational Cycles', 'unit': 'events', 'category': 'Compressor & Governor'},
    'bg_000': {'name': 'Transmission Clutch Actuator Pressure', 'unit': 'kPa', 'category': 'Auxiliary & Chassis'},
    'an_000': {'name': 'Air Dryer Desiccant Moisture Level', 'unit': 'ppm eq', 'category': 'Auxiliary & Chassis'},
    'bx_000': {'name': 'Trailer Brake Line Signal Pressure', 'unit': 'kPa', 'category': 'Braking Circuit'},
    'ao_000': {'name': 'Compressor Thermal Sensor / Engine Load', 'unit': 'units', 'category': 'Compressor & Governor'},
    'cs_002': {'name': 'Air Mass Flow Distribution Bin 2', 'unit': 'histogram bin', 'category': 'Auxiliary & Chassis'}
}

# Read real samples from test set
samples = []
with open("aps_failure_test_set.csv", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("class,"):
            headers = [h.strip() for h in line.split(",")]
            break
    reader = csv.DictReader(f, fieldnames=headers)
    for row in reader:
        samples.append(row)
        if len(samples) >= 300:
            break

# Separate pos and neg
pos_rows = [r for r in samples if r.get("class", "").strip() == "pos"]
neg_rows = [r for r in samples if r.get("class", "").strip() == "neg"]

preset_trucks = {
    "healthy": {
        "title": "Healthy Fleet Truck",
        "description": "Standard operational parameters with normal air leakage rates and optimal reservoir pressures.",
        "expected": "Healthy (neg)",
        "features": {feat: (float(neg_rows[0][feat]) if neg_rows[0].get(feat, "na") not in ["na", "", "nan"] else None) for feat in top_features}
    },
    "critical": {
        "title": "Imminent APS Failure",
        "description": "Severe pneumatic pressure loss, excessive compressor duty cycle, and high regeneration pulses.",
        "expected": "APS Failure (pos)",
        "features": {feat: (float(pos_rows[0][feat]) if pos_rows[0].get(feat, "na") not in ["na", "", "nan"] else None) for feat in top_features}
    },
    "borderline": {
        "title": "Elevated Wear & Tear",
        "description": "Moderate degradation in brake differential pressure and elevated moisture in desiccant dryer.",
        "expected": "Needs Inspection",
        "features": {feat: (float(pos_rows[1][feat]) if pos_rows[1].get(feat, "na") not in ["na", "", "nan"] else None) for feat in top_features}
    },
    "missing_sensors": {
        "title": "Partial Telemetry (Missing Values)",
        "description": "Simulates real-world sensor dropout where several sensors return 'na' values.",
        "expected": "Handled via Imputer",
        "features": {feat: ((None if i % 4 == 0 else float(neg_rows[1][feat])) if neg_rows[1].get(feat, "na") not in ["na", "", "nan"] else None) for i, feat in enumerate(top_features)}
    }
}

os.makedirs("webapp/static/data", exist_ok=True)
with open("webapp/static/data/presets.json", "w", encoding="utf-8") as f:
    json.dump({"meta": sensor_meta, "top_features": top_features, "presets": preset_trucks}, f, indent=2)

# Generate batch CSV with 15 mixed trucks for sample testing
batch_rows = []
for i in range(15):
    is_pos = (i in [2, 5, 8, 11])
    src = pos_rows[i % len(pos_rows)] if is_pos else neg_rows[i % len(neg_rows)]
    truck_id = f"SCAN-TK-{1000 + i}"
    row_dict = {"truck_id": truck_id}
    for feat in top_features:
        row_dict[feat] = src.get(feat, "na")
    batch_rows.append(row_dict)

with open("webapp/static/data/batch_sample_scania_trucks.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["truck_id"] + top_features)
    writer.writeheader()
    writer.writerows(batch_rows)

print("Generated presets.json and batch_sample_scania_trucks.csv successfully!")
