# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:07:06 2025

@author: Raynard
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report

class RFModel:
    def __init__(self):
        self.model = None

    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df.select_dtypes(include=['number'])
        y = df["Condition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        self.model = RandomForestClassifier(
            n_estimators=250,
            min_samples_split=4,
            min_samples_leaf=1,
            criterion='entropy',
            max_depth= None,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        cv_f1 = cross_val_score(self.model, X_train, y_train, cv=3, scoring='f1_macro').mean()
        test_f1 = f1_score(y_test, y_pred, average='macro')

        print(f"[RF] CV_f1={cv_f1:.3f}, TestSplit_f1={test_f1:.3f}")
        print(classification_report(y_test, y_pred))
        return self.model, test_f1
