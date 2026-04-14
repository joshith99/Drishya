#!/usr/bin/env python3
"""Inspect the YouTube-UGC MOS Excel file structure"""
import pandas as pd

path = '/workspace/datasets/youtube_ugc/MOS_for_YouTube_UGC_dataset.xlsx'

# Read all sheets
xl = pd.ExcelFile(path, engine='openpyxl')
print(f"Sheets: {xl.sheet_names}\n")

for sheet in xl.sheet_names:
    print(f"{'='*60}")
    print(f"Sheet: '{sheet}'")
    # Read with no header first to see raw rows
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine='openpyxl')
    print(f"Shape: {raw.shape}")
    print("First 10 rows:")
    print(raw.head(10).to_string())
    print()
