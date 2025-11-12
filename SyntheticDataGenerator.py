# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 23:15:58 2025

@author: Raynard
"""

"""
SyntheticDataGenerator_v6.py
Rule-based, per-condition evolutionary synthetic data generator.
Outputs columns:
Time,Current (A),Voltage (V),Wattage (W),Temperature Indoor (°C),
Temperature Outdoor (°C),Set Point (°C),Alpha (°C),Supply (°C),Return (°C),
Delta (°C),Beta (°C),Cycle,Unit,Condition
"""

import pandas as pd
import numpy as np
import os
import random

class SyntheticDataGenerator:
    def __init__(self, baseline_path):
        self.baseline = pd.read_csv(baseline_path)
        print(f"[SyntheticDataGenerator] Baseline loaded: {self.baseline.shape}")

    def evolve_dataset(self, params, output_path):
        """Generate new dataset variant based on evolution parameters."""
        df = self.baseline.copy()

        # Apply evolution transformations
        noise_amp = params.get("fault_int", 1.0)
        rule_exp = params.get("rule_exp", 1.0)

        # Add slight nonlinear transformation to simulate new domain
        for col in ["Current_scaled", "Voltage_scaled", "Wattage_scaled", "Alpha_scaled", "Beta_scaled", "Delta_scaled", 
                    'Current (A)', 'Voltage (V)', 'Wattage (W)', 'Alpha (°C)', 'Beta (°C)', 'Delta (°C)', 'Supply (°C)', 'Return (°C)',]:
            if col in df.columns:
                df[col] = df[col] * rule_exp + np.random.normal(0, 0.02 * noise_amp, size=len(df))

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False)
        print(f"[SyntheticDataGenerator] Saved evolved dataset ({len(df)} rows) at: {output_path}")
        return df

