# NASA Exoplanet Classification Research

This repository contains code and data for exoplanet candidate classification using ensemble models, RWKV, and Transformer architectures. The project leverages datasets from KOI, K2, and TOI missions, with comprehensive feature engineering and model evaluation.

## Datasets

| Dataset  | Description                        | File Location                                 | Link                                                                 |
|----------|------------------------------------|-----------------------------------------------|---------------------------------------------------------------------|
| KOI      | Kepler Object of Interest          | `dataset/KOI_*.csv`                           | [KOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) |
| K2       | Second Kepler Mission              | `dataset/k2pandc_*.csv`                       | [K2 Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc) |
| TOI      | TESS Object of Interest            | `dataset/TOI_*.csv`                           | [TOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI) |
| Kepler   | Kepler labelled time series data   | `dataset/ExoTest.csv`, `dataset/ExoTrain.csv` | [Kepler Kaggle](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data/) |

> **⚠️ Important:**
> Please download the Kepler Kaggle dataset from [here](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data/) and place the files (`ExoTest.csv`, `ExoTrain.csv`) into the `dataset` folder before running the code.

## Features Used

### KOI (59 features)
| Feature Name         | ... | Feature Name         |
|---------------------|-----|---------------------|
| kepid               | ... | koi_dikco_msky      |
| koi_fpflag_nt       | ... | koi_dikco_mra       |
| koi_fpflag_ss       | ... | koi_dikco_mdec      |
| koi_fpflag_co       | ... | koi_dicco_msky      |
| koi_fpflag_ec       | ... | koi_dicco_mra       |
| koi_period          | ... | koi_dicco_mdec      |
| koi_time0bk         | ... | koi_depth           |
| koi_time0           | ... | koi_prad            |
| koi_eccen           | ... | koi_sma             |
| koi_impact          | ... | koi_incl            |
| koi_duration        | ... | koi_teq             |
| koi_ror             | ... | koi_insol           |
| koi_srho            | ... | koi_dor             |
| koi_count           | ... | koi_model_snr       |
| koi_num_transits    | ... | koi_max_sngle_ev    |
| koi_tce_plnt_num    | ... | koi_max_mult_ev     |
| koi_bin_oedp_sig    | ... | koi_steff           |
| koi_slogg           | ... | koi_smet            |
| koi_srad            | ... | koi_smass           |
| ra                  | ... | dec                 |
| koi_kepmag          | ... | koi_gmag            |
| koi_rmag            | ... | koi_imag            |
| koi_zmag            | ... | koi_jmag            |
| koi_hmag            | ... | koi_kmag            |
| koi_fwm_stat_sig    | ... | koi_fwm_sra         |
| koi_fwm_sdec        | ... | koi_fwm_srao        |
| koi_fwm_sdeco       | ... | koi_fwm_prao        |
| koi_fwm_pdeco       | ... |                     |

### K2 (94 features)
| Feature Name         | ... | Feature Name         |
|---------------------|-----|---------------------|
| default_flag        | ... | st_nphot            |
| sy_snum             | ... | st_nrvc             |
| sy_pnum             | ... | st_nspec            |
| sy_mnum             | ... | pl_nespec           |
| cb_flag             | ... | pl_ntranspec        |
| disc_year           | ... | pl_ndispec          |
| rv_flag             | ... |                     |
| pul_flag            | ... |                     |
| ptv_flag            | ... |                     |
| tran_flag           | ... |                     |
| ast_flag            | ... |                     |
| obm_flag            | ... |                     |
| micro_flag          | ... |                     |
| etv_flag            | ... |                     |
| ima_flag            | ... |                     |
| dkin_flag           | ... |                     |
| pl_controv_flag     | ... |                     |
| pl_orbper           | ... |                     |
| pl_orbsmax          | ... |                     |
| pl_rade             | ... |                     |
| pl_radj             | ... |                     |
| pl_masse            | ... |                     |
| pl_massj            | ... |                     |
| pl_msinie           | ... |                     |
| pl_msinij           | ... |                     |
| pl_cmasse           | ... |                     |
| pl_cmassj           | ... |                     |
| pl_bmasse           | ... |                     |
| pl_bmassj           | ... |                     |
| pl_dens             | ... |                     |
| pl_orbeccen         | ... |                     |
| pl_insol            | ... |                     |
| pl_eqt              | ... |                     |
| pl_orbincl          | ... |                     |
| pl_tranmid          | ... |                     |
| ttv_flag            | ... |                     |
| pl_imppar           | ... |                     |
| pl_trandep          | ... |                     |
| pl_trandur          | ... |                     |
| pl_ratdor           | ... |                     |
| pl_ratror           | ... |                     |
| pl_occdep           | ... |                     |
| pl_orbtper          | ... |                     |
| pl_orblper          | ... |                     |
| pl_rvamp            | ... |                     |
| pl_projobliq        | ... |                     |
| pl_trueobliq        | ... |                     |
| st_teff             | ... |                     |
| st_rad              | ... |                     |
| st_mass             | ... |                     |
| st_met              | ... |                     |
| st_lum              | ... |                     |
| st_logg             | ... |                     |
| st_age              | ... |                     |
| st_dens             | ... |                     |
| st_vsin             | ... |                     |
| st_rotp             | ... |                     |
| st_radv             | ... |                     |
| ra                  | ... |                     |
| dec                 | ... |                     |
| glat                | ... |                     |
| glon                | ... |                     |
| elat                | ... |                     |
| elon                | ... |                     |
| sy_pm               | ... |                     |
| sy_pmra             | ... |                     |
| sy_pmdec            | ... |                     |
| sy_dist             | ... |                     |
| sy_plx              | ... |                     |
| sy_bmag             | ... |                     |
| sy_vmag             | ... |                     |
| sy_jmag             | ... |                     |
| sy_hmag             | ... |                     |
| sy_kmag             | ... |                     |
| sy_umag             | ... |                     |
| sy_gmag             | ... |                     |
| sy_rmag             | ... |                     |
| sy_imag             | ... |                     |
| sy_zmag             | ... |                     |
| sy_w1mag            | ... |                     |
| sy_w2mag            | ... |                     |
| sy_w3mag            | ... |                     |
| sy_w4mag            | ... |                     |
| sy_gaiamag          | ... |                     |
| sy_tmag             | ... |                     |
| sy_kepmag           | ... |                     |
| pl_nnotes           | ... |                     |
| k2_campaigns_num    | ... |                     |

### TOI (19 features)
| Feature Name         |
|---------------------|
| tid                 |
| ctoi_alias          |
| pl_pnum             |
| ra                  |
| dec                 |
| st_pmra             |
| st_pmdec            |
| pl_tranmid          |
| pl_orbper           |
| pl_trandurh         |
| pl_trandep          |
| pl_rade             |
| pl_insol            |
| pl_eqt              |
| st_tmag             |
| st_dist             |
| st_teff             |
| st_logg             |
| st_rad              |

## Model Performance

| Model       | Dataset | Accuracy | ROC AUC | Threshold | Notes |
|-------------|---------|----------|---------|-----------|-------|
| Ensemble    | KOI     | 0.9415   | -       | -         |       |
| Ensemble    | K2      | 0.9725   | -       | -         |       |
| Ensemble    | TOI     | 0.7093   | -       | -         |       |
| **RWKV_v7** | KOI     | 0.9894   | 0.9947  | 0.7183    | Raw Timeseries |
| [Timer Transformer](https://github.com/thuml/Large-Time-Series-Model) | KOI | 0.0071 | 0.5000 | 0.5996 | Raw Timeseries |

## Classification Reports

### RWKV (KOI)
- **ROC AUC Score (Optimal Threshold):** 0.9947
- **Accuracy (Optimal Threshold 0.7183):** 0.9894

### Transformer (KOI)
- **ROC AUC Score (Optimal Threshold):** 0.5000
- **Accuracy (Optimal Threshold 0.5996):** 0.0071

## Usage

1. Clone this repository to your local machine:
	```
	git clone https://github.com/gasthell/NoRush-A-World-Away-Hunting-for-Exoplanets-with-AI.git
	```
2. Install required dependencies:
	```
	pip install -r requirements.txt
	```
3. Download the Kepler Kaggle dataset and place `ExoTest.csv` and `ExoTrain.csv` in the `dataset` folder (see instructions above).
4. Launch the interactive interface:
	```
	python gradio_interface.py
	```
5. Explore the provided Jupyter notebooks for data analysis and model training workflows.

> Or you can use Live Demo!
> 
> Curious to see the model in action? Explore the live demo to use the prediction and analysis features.
> 
> Please note: This is a demonstration version. Advanced functionalities like model training and hyperparameter optimization are available in the full version, which can be run locally from the repository.
> 
> 👉 Launch the Live Demo: [norush.ai-archive-project.com](https://norush.ai-archive-project.com/) |

## Team & Hackathon Info

- Project developed for NASA Exoplanet Hackathon 2025
