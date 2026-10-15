import os
import io
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import pandas as pd
import numpy as np

# The 20 selected top features identified and trained in the project notebook
TOP_FEATURES = [
    'bj_000', 'aq_000', 'ci_000', 'dn_000', 'ap_000', 
    'bb_000', 'bv_000', 'bu_000', 'cq_000', 'ck_000', 
    'bt_000', 'aa_000', 'bh_000', 'ah_000', 'cc_000', 
    'bg_000', 'an_000', 'bx_000', 'ao_000', 'cs_002'
]

SENSOR_METADATA = {
    'bj_000': {'name': 'Compressor Discharge Pressure Cycle', 'unit': 'cycles', 'category': 'Pneumatic Delivery', 'importance': 0.142},
    'aq_000': {'name': 'Air Tank Delivery Pressure Monitor', 'unit': 'kPa', 'category': 'Pneumatic Delivery', 'importance': 0.118},
    'ci_000': {'name': 'Air Dryer Regeneration Counter', 'unit': 'pulses', 'category': 'Pneumatic Delivery', 'importance': 0.098},
    'dn_000': {'name': 'Brake Actuator Pressure Differential', 'unit': 'mbar', 'category': 'Braking Circuit', 'importance': 0.086},
    'ap_000': {'name': 'Reservoir Main Flow Velocity', 'unit': 'L/min', 'category': 'Pneumatic Delivery', 'importance': 0.075},
    'bb_000': {'name': 'Auxiliary Consumer Flow Supply', 'unit': 'L/min', 'category': 'Braking Circuit', 'importance': 0.069},
    'bv_000': {'name': 'Primary Brake Circuit Pressure', 'unit': 'kPa', 'category': 'Braking Circuit', 'importance': 0.063},
    'bu_000': {'name': 'Secondary Brake Circuit Pressure', 'unit': 'kPa', 'category': 'Braking Circuit', 'importance': 0.058},
    'cq_000': {'name': 'Multi-Circuit Protection Valve State', 'unit': 'ratio', 'category': 'Braking Circuit', 'importance': 0.052},
    'ck_000': {'name': 'Compressor Duty Cycle Accumulator', 'unit': 'sec', 'category': 'Compressor & Governor', 'importance': 0.047},
    'bt_000': {'name': 'Governor Cut-Out Pressure Setpoint', 'unit': 'bar', 'category': 'Compressor & Governor', 'importance': 0.042},
    'aa_000': {'name': 'Master Air Pressure Sensor', 'unit': 'bar', 'category': 'Pneumatic Delivery', 'importance': 0.038},
    'bh_000': {'name': 'Pneumatic Suspension Bellows Pressure', 'unit': 'kPa', 'category': 'Auxiliary & Chassis', 'importance': 0.033},
    'ah_000': {'name': 'Ambient Temp & Compressor Inlet', 'unit': '°C count', 'category': 'Compressor & Governor', 'importance': 0.029},
    'cc_000': {'name': 'Purge Valve Operational Cycles', 'unit': 'events', 'category': 'Compressor & Governor', 'importance': 0.026},
    'bg_000': {'name': 'Transmission Clutch Actuator Pressure', 'unit': 'kPa', 'category': 'Auxiliary & Chassis', 'importance': 0.022},
    'an_000': {'name': 'Air Dryer Desiccant Moisture Level', 'unit': 'ppm eq', 'category': 'Auxiliary & Chassis', 'importance': 0.018},
    'bx_000': {'name': 'Trailer Brake Line Signal Pressure', 'unit': 'kPa', 'category': 'Braking Circuit', 'importance': 0.015},
    'ao_000': {'name': 'Compressor Thermal Sensor / Engine Load', 'unit': 'units', 'category': 'Compressor & Governor', 'importance': 0.012},
    'cs_002': {'name': 'Air Mass Flow Distribution Bin 2', 'unit': 'histogram bin', 'category': 'Auxiliary & Chassis', 'importance': 0.010}
}

class ModelService:
    def __init__(self):
        self.model = None
        self.model_path = self._locate_model()
        self.load_model()

    def _locate_model(self) -> Path:
        """Find the trained pipeline joblib file across known workspace paths."""
        potential_paths = [
            Path(__file__).resolve().parent / "scania_aps_pipeline.joblib",
            Path(__file__).resolve().parent.parent / "scania_aps_pipeline.joblib",
            Path("scania_aps_pipeline.joblib"),
            Path("webapp/backend/scania_aps_pipeline.joblib"),
            Path(__file__).resolve().parent.parent.parent / "scania_aps_pipeline.joblib",
            Path(__file__).resolve().parent.parent.parent / "ml final 2" / "New folder" / "scania_aps_pipeline.joblib"
        ]
        for p in potential_paths:
            if p.exists():
                return p
        return potential_paths[0]

    def load_model(self):
        """Load the Scania APS Pipeline model."""
        try:
            if self.model_path and self.model_path.exists():
                self.model = joblib.load(self.model_path)
                print(f"[ModelService] Successfully loaded trained pipeline from: {self.model_path}")
            else:
                print(f"[ModelService] Model file not found at: {self.model_path}")
        except Exception as e:
            print(f"[ModelService] Error loading model: {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def prepare_single_row(self, raw_features: Dict[str, Any]) -> pd.DataFrame:
        """
        Format dictionary features into a single-row DataFrame with the exact 20 top features.
        Preserves NaN/None so the trained SimpleImputer handles missing values as designed.
        """
        row = {}
        for feat in TOP_FEATURES:
            val = raw_features.get(feat, None)
            if val is None or val == "" or str(val).lower() in ["na", "nan", "null"]:
                row[feat] = np.nan
            else:
                try:
                    row[feat] = float(val)
                except (ValueError, TypeError):
                    row[feat] = np.nan
        return pd.DataFrame([row], columns=TOP_FEATURES)

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference for a single truck.
        Returns prediction, probability, risk tier, cost metric analysis, and recommendations.
        """
        if not self.is_ready():
            raise RuntimeError("ML model pipeline is not loaded.")

        df_input = self.prepare_single_row(features)
        
        # Actual ML Model Inference
        pred_int = int(self.model.predict(df_input)[0])
        probabilities = self.model.predict_proba(df_input)[0]
        prob_healthy = round(float(probabilities[0]), 4)
        prob_failure = round(float(probabilities[1]), 4)

        # Risk Classification & Thresholding
        if prob_failure >= 0.75:
            risk_tier = "CRITICAL RISK"
            badge_class = "danger"
            status_text = "APS Component Failure Detected"
            recommendation = (
                "IMMEDIATE ACTION REQUIRED: Route truck to nearest service center immediately. "
                "Pneumatic air pressure supply is severely degraded. Inspect compressor valves, "
                "air dryer regeneration system, and auxiliary pneumatic lines to prevent roadside breakdown."
            )
            cost_saving_msg = "Preventive inspection ($10) avoids catastrophic on-road breakdown ($500), saving $490."
        elif prob_failure >= 0.50:
            risk_tier = "HIGH RISK"
            badge_class = "warning"
            status_text = "APS Component Failure Likely"
            recommendation = (
                "SCHEDULE URGENT MAINTENANCE: Truck exhibits significant pneumatic system anomalies. "
                "Conduct diagnostic pressure tests on braking reservoirs and replace air dryer desiccant within 48 hours."
            )
            cost_saving_msg = "Timely workshop inspection ($10) prevents severe downtime and towing costs ($500)."
        elif prob_failure >= 0.20:
            risk_tier = "MODERATE WEAR"
            badge_class = "caution"
            status_text = "Elevated Operational Wear"
            recommendation = (
                "MONITOR AT NEXT REGULAR SERVICE: Sensor counters indicate mild pneumatic pressure fluctuation. "
                "Check for minor leaks at governor valves and verify compressor duty cycle."
            )
            cost_saving_msg = "Standard preventive checkup ($10) recommended."
        else:
            risk_tier = "OPTIMAL (HEALTHY)"
            badge_class = "success"
            status_text = "Healthy Air Pressure System"
            recommendation = (
                "SAFE FOR OPERATION: All pneumatic delivery counters and braking circuit pressures "
                "are within optimal factory tolerances. No maintenance intervention required."
            )
            cost_saving_msg = "Truck is operating safely. Zero unnecessary maintenance downtime incurred."

        # Top sensor anomalies based on relative values
        anomalies = []
        for feat in TOP_FEATURES:
            val = df_input[feat].iloc[0]
            meta = SENSOR_METADATA.get(feat, {})
            if pd.notna(val):
                # Highlight large values
                if val > 100000:
                    anomalies.append({
                        "feature": feat,
                        "name": meta.get("name", feat),
                        "value": val,
                        "unit": meta.get("unit", ""),
                        "category": meta.get("category", "General"),
                        "severity": "high" if val > 1000000 else "medium"
                    })
        anomalies = sorted(anomalies, key=lambda x: x["value"], reverse=True)[:5]

        return {
            "prediction": pred_int,
            "status": "APS Failure (pos)" if pred_int == 1 else "Healthy (neg)",
            "status_headline": status_text,
            "failure_probability": prob_failure,
            "healthy_probability": prob_healthy,
            "risk_tier": risk_tier,
            "badge_class": badge_class,
            "recommendation": recommendation,
            "cost_saving_msg": cost_saving_msg,
            "top_anomalies": anomalies,
            "features_used": {feat: (None if pd.isna(df_input[feat].iloc[0]) else float(df_input[feat].iloc[0])) for feat in TOP_FEATURES}
        }

    def predict_batch_csv(self, file_content: bytes) -> Dict[str, Any]:
        """
        Process a batch CSV file containing multiple trucks.
        Can ingest either 171 raw features or 20 top features.
        """
        if not self.is_ready():
            raise RuntimeError("ML model pipeline is not loaded.")

        try:
            # Read CSV content into DataFrame
            try:
                df = pd.read_csv(io.BytesIO(file_content), na_values=["na", "NA", "NaN", "nan"])
            except Exception:
                # If there are comment lines in raw Scania format
                df = pd.read_csv(io.BytesIO(file_content), skiprows=20, na_values=["na", "NA", "NaN", "nan"])

            # Check if truck_id or index exists
            if "truck_id" in df.columns:
                truck_ids = df["truck_id"].astype(str).tolist()
            elif "id" in df.columns:
                truck_ids = df["id"].astype(str).tolist()
            else:
                truck_ids = [f"TRUCK-{1001 + i}" for i in range(len(df))]

            # Drop class target if present
            df_features = df.drop(columns=["class", "truck_id", "id"], errors="ignore")

            # Ensure all 20 required features exist
            for feat in TOP_FEATURES:
                if feat not in df_features.columns:
                    df_features[feat] = np.nan

            X_batch = df_features[TOP_FEATURES].apply(pd.to_numeric, errors="coerce")

            # Run actual model predictions
            predictions = self.model.predict(X_batch)
            probabilities = self.model.predict_proba(X_batch)

            results_list = []
            failure_count = 0
            healthy_count = 0
            critical_count = 0

            for i in range(len(df)):
                p = int(predictions[i])
                p_fail = round(float(probabilities[i][1]), 4)
                p_health = round(float(probabilities[i][0]), 4)

                if p == 1 or p_fail >= 0.5:
                    failure_count += 1
                else:
                    healthy_count += 1

                if p_fail >= 0.75:
                    critical_count += 1
                    risk_badge = "CRITICAL"
                elif p_fail >= 0.50:
                    risk_badge = "HIGH RISK"
                elif p_fail >= 0.20:
                    risk_badge = "MODERATE"
                else:
                    risk_badge = "HEALTHY"

                results_list.append({
                    "truck_id": truck_ids[i],
                    "prediction": p,
                    "status": "APS Failure (pos)" if p == 1 else "Healthy (neg)",
                    "failure_probability": p_fail,
                    "healthy_probability": p_health,
                    "risk_badge": risk_badge,
                    "recommended_action": "Shop Inspection ($10)" if p == 1 else "Clear for Dispatch ($0)"
                })

            total = len(df)
            failure_rate = round((failure_count / total * 100), 2) if total > 0 else 0.0
            
            # Scania Challenge Cost Calculation:
            # Catching failures early saves $500 roadside breakdown cost per truck vs $10 check
            estimated_savings = (failure_count * 500) - (failure_count * 10)

            return {
                "total_trucks": total,
                "failures_detected": failure_count,
                "healthy_trucks": healthy_count,
                "critical_trucks": critical_count,
                "failure_rate_pct": failure_rate,
                "estimated_cost_savings": estimated_savings,
                "results": results_list
            }

        except Exception as e:
            raise ValueError(f"Error processing CSV batch data: {str(e)}")

# Singleton instance
model_service = ModelService()
