import pandas as pd
import lightkurve as lk
import numpy as np
import csv
import os
import time
from tqdm import tqdm
import warnings

INPUT_CSV = 'dataset/cumulative_2025.10.04_09.09.34.csv'
OUTPUT_CSV = 'exoplanet_dataset_sequential.csv'
N_BINS = 201

warnings.filterwarnings('ignore', category=lk.LightkurveWarning)
from astropy.units import UnitsWarning
warnings.filterwarnings('ignore', category=UnitsWarning, append=True)
from astropy.io.fits.verify import VerifyWarning
warnings.filterwarnings('ignore', category=VerifyWarning, append=True)

def get_lightcurve(kepid):
    """Downloads and prepares a light curve for a single Kepler ID."""
    try:
        search_result = lk.search_lightcurve(f"KIC {kepid}")
        if not search_result:
            return None
        lc_collection = search_result.download_all()
        if not lc_collection:
            return None
        return lc_collection.stitch().flatten().remove_outliers(sigma=5)
    except Exception as e:
        print(f"      [FAIL] KIC {kepid}: Download/Stitch failed. Reason: {e}")
        return None

def process_lightcurve(lc, kepid, period, n_bins=N_BINS):
    """Folds and bins a light curve into a fixed-length sequence."""
    if lc is None or period is None or np.isnan(period) or period <= 0:
        return None
    try:
        folded_lc = lc.fold(period=period)
        binned_lc = folded_lc.bin(bins=n_bins)
        normalized_flux = binned_lc.flux.value / np.nanmedian(binned_lc.flux.value)
        if np.isnan(normalized_flux).any():
            normalized_flux = pd.Series(normalized_flux).fillna(1.0).values
        if len(normalized_flux) == n_bins:
            return normalized_flux
        else:
            return None
    except Exception as e:
        print(f"      [FAIL] KIC {kepid}: Processing failed for period {period}. Reason: {e}")
        return None

def create_dataset_sequentially():
    """Main function to orchestrate the dataset creation process sequentially."""
    print("--- 1. Loading and Preparing Metadata ---")
    df = pd.read_csv(INPUT_CSV, comment='#')
    df = df[0:200]
    df = df[['kepid', 'kepoi_name', 'koi_period', 'koi_disposition']]
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    df['LABEL'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Found {len(df)} total samples to process.")

    processed_kepids = set()
    header = ['kepid', 'kepoi_name', 'LABEL'] + [f'FLUX.{i+1}' for i in range(N_BINS)]
    
    if os.path.exists(OUTPUT_CSV):
        print(f"--- Found existing file: '{OUTPUT_CSV}'. Resuming download. ---")
        with open(OUTPUT_CSV, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                processed_kepids.add(int(row[0]))
        print(f"Found {len(processed_kepids)} stars already processed. Skipping them.")
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    print(f"\n--- Starting processing loop. Writing successful results to '{OUTPUT_CSV}' ---")
    
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Stars"):
            kepid = row['kepid']
            
            if kepid in processed_kepids:
                continue
            
            light_curve = get_lightcurve(kepid)
            sequence = process_lightcurve(light_curve, kepid, row['koi_period'])
            
            if sequence is not None:
                flux_dict = {f'FLUX.{i+1}': val for i, val in enumerate(sequence)}
                output_row = {
                    'kepid': row['kepid'],
                    'kepoi_name': row['kepoi_name'],
                    'LABEL': row['LABEL'],
                    **flux_dict
                }
                writer.writerow(output_row)
            
            time.sleep(0.1)

    print(f"\n--- 4. SUCCESS! ---")
    print(f"Dataset creation complete. File saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    create_dataset_sequentially()