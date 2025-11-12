"""
import pandas as pd
from sklearn.metrics import f1_score, classification_report

class RFValidator:
    def __init__(self, real_validation_path):
        self.real_data = pd.read_csv(real_validation_path)
        print(f"[Validator] Loaded real validation dataset: {self.real_data.shape}")

    def validate(self, model, feature_reference_df=None):
        """
#        Validates model on real data.
#        Ensures feature alignment with training data.
#        """
"""
        X_real = self.real_data.select_dtypes(include=['number'])
        y_real = self.real_data["Condition"]

        # === Feature alignment ===
        if feature_reference_df is not None:
            # Gunakan kolom numerik dari dataset pelatihan
            feature_cols = feature_reference_df.select_dtypes(include=['number']).columns
            # Tambahkan kolom yang hilang di real data
            for col in feature_cols:
                if col not in X_real.columns:
                    X_real[col] = 0.0
            # Hapus kolom ekstra di real data
            X_real = X_real[feature_cols]

        # === Prediction & Evaluation ===
        y_pred = model.predict(X_real)
        f1_macro = f1_score(y_real, y_pred, average='weighted')
        print(f"[Validator_v2] Real F1_macro={f1_macro:.3f}")
        print(classification_report(y_real, y_pred))
        return f1_macro
"""
#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class RFValidator:
    def __init__(self, real_validation_path, output_dir=None):
        """
        Initialize the validator by loading real-world validation data.
        Args:
            real_validation_path (str): Path to the real validation CSV file.
            output_dir (str): Optional directory to save best confusion matrix.
        """
        self.real_data = pd.read_csv(real_validation_path)
        self.output_dir = output_dir
        print(f"[Validator] Loaded real validation dataset: {self.real_data.shape}")

        if "Condition" not in self.real_data.columns:
            raise ValueError("Real validation data must contain a 'Condition' column.")

    def validate(self, model, feature_reference_df=None, save_best=False, normalize=False):
        """
        Validate the trained model on the real dataset.
        Generates classification metrics and optionally saves confusion matrix for the best model.

        Args:
            model: Trained scikit-learn classifier (e.g., RandomForest).
            feature_reference_df: DataFrame of training (synthetic) features.
            save_best (bool): If True, save confusion matrix as 'confusion_matrix_best_model.png'.
            normalize (bool): If True, plot confusion matrix as normalized (percentage).

        Returns:
            dict: Full classification report for downstream optimization.
        """
        # --- Prepare Real Data ---
        X_real = self.real_data.select_dtypes(include=['number']).copy()
        y_real = self.real_data["Condition"]

        # === Feature alignment ===
        if feature_reference_df is not None:
            feature_cols = feature_reference_df.select_dtypes(include=['number']).columns
            # Add missing columns
            for col in feature_cols:
                if col not in X_real.columns:
                    X_real[col] = 0.0
            # Remove extra columns
            X_real = X_real[feature_cols]

        # --- Prediction ---
        y_pred = model.predict(X_real)

        # --- Classification Report ---
        report = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
        f1_weighted = report['weighted avg']['f1-score']
        print(f"[Validator] Real F1_weighted={f1_weighted:.3f}")
        print(classification_report(y_real, y_pred, zero_division=0))

        # --- Save Confusion Matrix ONLY if it's the best model ---
        if save_best:
            try:
                labels = sorted(list(set(y_real.unique()) | set(y_pred)))
                cm = confusion_matrix(y_real, y_pred, labels=labels)

                if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
                    fmt = ".2f"
                    title_suffix = "(Normalized)"
                else:
                    fmt = "d"
                    title_suffix = "(Counts)"

                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                            xticklabels=labels, yticklabels=labels)
                plt.title(f"Confusion Matrix {title_suffix} – Best Model on Real Data")
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.tight_layout()

                if self.output_dir:
                    save_path = os.path.join(self.output_dir, "confusion_matrix_best_model.png")
                    plt.savefig(save_path, dpi=300)
                    print(f"[Validator] 🖼 Confusion matrix (best model) saved to: {save_path}")

                plt.close()
            except Exception as e:
                print(f"[Validator] ⚠️ Could not plot/save confusion matrix: {e}")

        # Return the classification report dictionary
        return report