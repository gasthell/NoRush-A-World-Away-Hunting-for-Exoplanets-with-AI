import gradio as gr
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import optuna
import glob

# ML Imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import KNNImputer
from itertools import cycle

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.amp import GradScaler, autocast
    # This requires the 'models' directory from your notebook to be in the same folder
    from models import TimerBackbone
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch or the 'models.TimerBackbone' module is not available. The Time Series Trainer tab will be disabled.")

import json
import gc
import time

# --- Global Settings ---
MODEL_DATA = {}
MODEL_CACHE_DIR = "saved_models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# --- Model Saving and Loading ---
def save_model_to_cache(dataset_name):
    if dataset_name in MODEL_DATA:
        data_to_save = MODEL_DATA[dataset_name].copy()
        data_to_save.pop('shap_explainer', None)
        
        save_path = os.path.join(MODEL_CACHE_DIR, f"{dataset_name}_model.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Model for '{dataset_name}' saved to cache at {save_path}")

# --- Data Loading and Preprocessing ---
def prepare_and_train(dataset_name, filepath, target_col, progress=gr.Progress()):
    global MODEL_DATA
    if not dataset_name.strip():
        return "Error: Please provide a name for your custom ensemble.", gr.Dropdown(choices=list(MODEL_DATA.keys()))

    progress(0, desc="Starting Training...")
    print(f"\n--- Preparing and training STACKING ENSEMBLE for: {dataset_name} ---")
    try:
        df = pd.read_csv(filepath.name if hasattr(filepath, 'name') else filepath, comment='#')
    except Exception as e: return f"Error reading CSV: {e}", gr.Dropdown(choices=list(MODEL_DATA.keys()))

    if target_col not in df.columns: return f"Error: Target column '{target_col}' not found.", gr.Dropdown(choices=list(MODEL_DATA.keys()))
    
    progress(0.1, desc="Cleaning & Preparing Data...")
    df.dropna(subset=[target_col], inplace=True)
    y = df[target_col].astype('category')
    y_codes = y.cat.codes
    X = df.drop(columns=[target_col, 'koi_disposition', 'koi_pdisposition', 'disposition', 'koi_score', 'toi', 'toipfx'], errors='ignore')
    X_numeric = X.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan)
    X_numeric.dropna(axis=1, how='all', inplace=True)
    
    feature_cols = list(X_numeric.columns)
    if not feature_cols: return "Error: No non-empty numeric columns found.", gr.Dropdown(choices=list(MODEL_DATA.keys()))

    progress(0.2, desc="Imputing Missing Values...")
    imputer = KNNImputer(n_neighbors=5).fit(X_numeric)
    X_processed = pd.DataFrame(imputer.transform(X_numeric), columns=feature_cols)

    scaler = StandardScaler().fit(X_processed)
    X_scaled = scaler.transform(X_processed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_codes, test_size=0.2, random_state=42, stratify=y_codes)
    
    progress(0.3, desc="Defining Base Models...")
    base_estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss')),
        ('lgbm', LGBMClassifier(random_state=42)),
        ('mlp', MLPClassifier(random_state=42, max_iter=1000)),
    ]
    
    progress(0.5, desc="Training Stacking Ensemble...")
    stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(), cv=3)
    stacking_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, stacking_model.predict(X_test))
    
    progress(0.8, desc="Calculating SHAP Values...")
    X_sample = pd.DataFrame(X_train, columns=feature_cols).sample(min(50, len(X_train)), random_state=42)
    shap_explainer = shap.KernelExplainer(stacking_model.predict_proba, X_sample)
    shap_values = shap_explainer.shap_values(X_sample)
    
    MODEL_DATA[dataset_name] = {
        'ensemble_model': stacking_model, 'scaler': scaler, 'imputer': imputer,
        'class_names': y.cat.categories, 'features': feature_cols, 'accuracy': accuracy,
        'shap_explainer': shap_explainer, 'shap_values': shap_values, 'X_sample': X_sample,
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test
    }
    
    if dataset_name in ["KOI", "K2", "TOI"]: save_model_to_cache(dataset_name)
    progress(1, desc="Training Complete!")
    return f"Ensemble '{dataset_name}' trained. Accuracy: {accuracy:.4f}", gr.Dropdown(choices=list(MODEL_DATA.keys()), value=dataset_name)

# --- Bayesian Optimization with Optuna ---
def bayesian_optimizer(dataset_name, n_trials=20, progress=gr.Progress()):
    if dataset_name not in MODEL_DATA: return "Dataset not found.", None, None, None, None
    data = MODEL_DATA[dataset_name]
    X_train, y_train = data['X_train'], data['y_train']

    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e2, log=True)
        model = StackingClassifier(
            estimators=[(name, model) for name, model in data['ensemble_model'].named_estimators.items()],
            final_estimator=LogisticRegression(C=C, random_state=42, max_iter=1000), cv=3)
        model.fit(X_train, y_train)
        return accuracy_score(data['y_test'], model.predict(data['X_test']))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[lambda study, trial: progress(trial.number / n_trials, desc=f"Trial {trial.number+1}/{n_trials}")])
    best_params = study.best_params

    best_model = StackingClassifier(
        estimators=[(name, model) for name, model in data['ensemble_model'].named_estimators.items()],
        final_estimator=LogisticRegression(C=best_params['C'], random_state=42, max_iter=1000), cv=3)
    best_model.fit(data['X_train'], data['y_train'])
    new_accuracy = accuracy_score(data['y_test'], best_model.predict(data['X_test']))

    MODEL_DATA[dataset_name]['ensemble_model'] = best_model
    MODEL_DATA[dataset_name]['accuracy'] = new_accuracy
    if dataset_name in ["KOI", "K2", "TOI"]: save_model_to_cache(dataset_name)
    
    log = f"Optimization finished. Best Params: {best_params}\nNew Accuracy: {new_accuracy:.4f}"
    report, fig_cm, fig_roc, fig_shap = get_full_model_details(dataset_name)
    return log, report, fig_cm, fig_roc, fig_shap

# --- Gradio UI Functions ---
def get_full_model_details(dataset_name):
    plt.close('all')
    if dataset_name not in MODEL_DATA: return "No model selected.", None, None, None

    data = MODEL_DATA[dataset_name]
    y_test = data['y_test']
    X_test_df = pd.DataFrame(data['X_test'], columns=data['features'])
    y_pred = data['ensemble_model'].predict(X_test_df)
    y_prob = data['ensemble_model'].predict_proba(X_test_df)
    class_names = data['class_names']
    
    features_md = "### Features Used (" + str(len(data['features'])) + " total)\n" + ", ".join(f"`{f}`" for f in data['features'])
    report = f"# Report: {dataset_name}\n**Ensemble Accuracy**: {data['accuracy']:.4f}\n\n" + features_md
    
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6)); disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=class_names); disp.plot(ax=ax_cm, cmap=plt.cm.Blues); ax_cm.set_title("Confusion Matrix"); plt.tight_layout()
    
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    if len(class_names) == 2:
        # --- Handle Binary Case ---
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    else:
        # --- Handle Multi-class Case ---
        y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
        for i, color in zip(range(len(class_names)), cycle(['aqua', 'darkorange', 'cornflowerblue'])):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (area = {roc_auc:0.2f})')
    
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2); ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05]); ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate'); ax_roc.set_title('ROC Curves'); ax_roc.legend(loc="lower right"); plt.tight_layout()

    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
    if 'shap_values' in data and data['shap_values'] is not None:
        if len(class_names) == 2:
            ax_shap.text(0.5, 0.5, 'Global SHAP plot not available.', ha='center', va='center')
        else:
            shap.summary_plot(data['shap_values'], data['X_sample'], plot_type="bar", show=False, class_names=class_names)
            ax_shap.set_title("Global SHAP Feature Importance")
            plt.tight_layout()
    else:
        ax_shap.text(0.5, 0.5, 'Global SHAP plot not available.', ha='center', va='center')
    
    return report, fig_cm, fig_roc, fig_shap

def predict_from_file(model_name, file, progress=gr.Progress()):
    if model_name not in MODEL_DATA: return None, "Please select a model."
    progress(0, desc="Starting Batch Prediction...")
    data = MODEL_DATA[model_name]
    ensemble, scaler, imputer, class_names, features = data['ensemble_model'], data['scaler'], data['imputer'], data['class_names'], data['features']
    try:
        progress(0.2, desc="Reading and cleaning CSV...")
        df = pd.read_csv(file.name, comment='#')
        df_predict = pd.DataFrame(columns=features)
        for col in features:
            if col in df.columns: df_predict[col] = df[col]
        df_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        progress(0.5, desc="Imputing and scaling data...")
        df_imputed = imputer.transform(df_predict)
        df_scaled = scaler.transform(df_imputed)
        
        progress(0.8, desc="Making predictions...")
        predictions = ensemble.predict(df_scaled)
        
        result_df = pd.DataFrame(df_imputed, columns=features)
        result_df['predicted_disposition'] = [class_names[p] for p in predictions]
        progress(1, desc="Done!")
        return result_df, "Prediction successful."
    except Exception as e: return None, f"An error occurred: {str(e)}"

def download_model(dataset_name):
    if dataset_name in MODEL_DATA:
        file_path = os.path.join(MODEL_CACHE_DIR, f"{dataset_name}_download.pkl")
        with open(file_path, 'wb') as f: pickle.dump(MODEL_DATA[dataset_name], f)
        return file_path
    return None

if TORCH_AVAILABLE:
    class ExoDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)
        def __len__(self): return len(self.features)
        def __getitem__(self, idx): return self.features[idx], self.labels[idx]

    class TimeSeriesModel(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.task_name = configs.task_name
            self.backbone = TimerBackbone.Model_RWKV7(configs)
            self.classification_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(configs.d_model, 1))
        def classification(self, x_enc):
            B, M, L = x_enc.shape
            means = x_enc.mean(2, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev
            enc_out, n_vars = self.backbone.patch_embedding(x_enc)
            enc_out, attns = self.backbone.decoder(enc_out)
            output = enc_out.reshape(B, M, -1, self.backbone.d_model).mean(dim=1).permute(0, 2, 1)
            return self.classification_head(output)
        def forward(self, x_enc, **kwargs):
            return self.classification(x_enc) if self.task_name == 'classification' else None

    class Configs:
        def __init__(self, config_dict):
            for key, value in config_dict.items(): setattr(self, key, value)

    def train_fold_gradio(fold_num, X_train_fold, y_train_fold, X_val_fold, y_val_fold, config, lr, prefix):
        log_accumulator = ""
        log_accumulator += f"\n{'='*25} FOLD {fold_num+1} {'='*25}\n"
        yield log_accumulator

        configs = Configs(config)
        model = TimeSeriesModel(configs).cuda()
        train_dataset = ExoDataset(X_train_fold, y_train_fold)
        val_dataset = ExoDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epoch'], eta_min=1e-6)
        
        counts = pd.Series(y_train_fold).value_counts()
        pos_weight_tensor = torch.tensor([counts.get(0, 1) / counts.get(1, 1)], dtype=torch.float32).cuda()
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        scaler = GradScaler()
        
        best_model_path = os.path.join(config['save_ckpt_path']+'_'+prefix, f"best_model_fold_{fold_num+1}.pth")
        best_val_auc = -1; patience_counter = 0

        for epoch in range(config['epoch']):
            model.train()
            all_preds, all_targets = [], []
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_log = f"Epoch {epoch+1}/{config['epoch']} (LR: {current_lr:.6f})\n"
            
            for i, batch in enumerate(train_loader):
                input_x, gt = batch[0].cuda(), batch[1].cuda().float().unsqueeze(1)
                with autocast("cuda"):
                    output = model(input_x)
                    loss = loss_function(output, gt)
                optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                all_preds.extend(torch.sigmoid(output).cpu().detach().numpy())
                all_targets.extend(gt.cpu().detach().numpy())
            
            scheduler.step()
            train_auc = roc_auc_score(all_targets, all_preds)
            
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_x, gt = batch[0].cuda(), batch[1].cuda().float().unsqueeze(1)
                    with autocast("cuda"): output = model(input_x)
                    val_preds.extend(torch.sigmoid(output).cpu().numpy())
                    val_targets.extend(gt.cpu().numpy())
            val_auc = roc_auc_score(val_targets, val_preds)

            epoch_log += f" --> Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
            log_accumulator += epoch_log + "\n"
            yield log_accumulator

            if val_auc > best_val_auc:
                best_val_auc = val_auc; torch.save(model.state_dict(), best_model_path); patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 5:
                log_accumulator += f"Early stopping triggered at epoch {epoch+1}.\n"
                yield log_accumulator
                break
        
        del model, train_loader, val_loader; gc.collect(); torch.cuda.empty_cache()
        yield best_val_auc

    def start_ts_training(file_path, epochs, batch_size, lr):
        log_stream = "Starting Time Series Model Training...\n"
        yield log_stream

        if file_path is None:
            yield "ERROR: Please upload a dataset file before starting training."
            return

        try:
            with open('config.json', 'r') as f: config = json.load(f)
            config['epoch'] = int(epochs)
            config['batch_size'] = int(batch_size)
        except FileNotFoundError:
            yield "ERROR: config.json not found. Please ensure it is in the root directory."
            return

        prefix = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())

        os.makedirs(config['save_ckpt_path'] + '_' + str(prefix), exist_ok=True)
        log_stream += "Configuration loaded.\n"
        yield log_stream
        
        try:
            log_stream += f"Loading dataset from {os.path.basename(file_path.name)}...\n"
            yield log_stream
            df_train = pd.read_csv(file_path.name)
            
            if 'LABEL' not in df_train.columns:
                yield "ERROR: The uploaded CSV must contain a 'LABEL' column."
                return
            
            if not np.all(np.isin(df_train['LABEL'], [1, 2])):
                 yield "ERROR: The 'LABEL' column must contain only values 1 (no exoplanet) and 2 (exoplanet)."
                 return

            X = df_train.drop('LABEL', axis=1).values
            y = (df_train['LABEL'] - 1).values
            X_prepared = np.expand_dims(X, axis=1)

        except Exception as e:
            yield f"ERROR: Could not read or process the uploaded file. Details: {str(e)}"
            return
            
        log_stream += "Dataset loaded and prepared successfully.\n"
        yield log_stream

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_scores = []
        full_log = log_stream

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_prepared, y)):
            X_train_fold, X_val_fold = X_prepared[train_idx], X_prepared[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            fold_generator = train_fold_gradio(fold, X_train_fold, y_train_fold, X_val_fold, y_val_fold, config, lr, str(prefix))
            
            current_fold_log = ""
            for output in fold_generator:
                if isinstance(output, str):
                    current_fold_log = output
                    yield full_log + current_fold_log
                else:
                    best_auc = output
                    oof_scores.append(best_auc)
            full_log += current_fold_log
        
        final_summary = f"\n{'='*25} CROSS-VALIDATION FINISHED {'='*25}\n"
        final_summary += f"Scores for each fold: {[f'{score:.4f}' for score in oof_scores]}\n"
        final_summary += f"Mean CV AUC: {np.mean(oof_scores):.4f}\n"
        final_summary += f"Std Dev CV AUC: {np.std(oof_scores):.4f}\n"
        full_log += final_summary
        yield full_log
    
    # --- New Functions for Time Series Prediction ---
    def get_model_folders(base_path="output_weight"):
        if not os.path.isdir(base_path):
            return []
        return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
    def run_ts_prediction(model_folder, prediction_file, progress=gr.Progress()):
        if not model_folder:
            return None, "Error: Please select a trained model folder."
        if prediction_file is None:
            return None, "Error: Please upload a CSV file for prediction."

        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            return None, "Error: config.json not found. Cannot proceed with inference."

        model_folder_path = os.path.join("output_weight", model_folder)
        model_files = glob.glob(os.path.join(model_folder_path, "*.pth"))

        if not model_files:
            return None, f"Error: No model files (.pth) found in {model_folder_path}."

        try:
            progress(0.1, desc="Reading CSV data...")
            df_pred = pd.read_csv(prediction_file.name)
            
            # --- Data Validation ---
            expected_cols = 3198
            if df_pred.shape[1] != expected_cols:
                return None, f"Error: The input CSV must have exactly {expected_cols} columns of flux data. Your file has {df_pred.shape[1]} columns."

            X_pred_data = df_pred.values
            X_pred_prepared = np.expand_dims(X_pred_data, axis=1)

        except Exception as e:
            return None, f"Error reading or processing CSV: {str(e)}"

        progress(0.3, desc="Preparing data loaders...")
        configs = Configs(config)
        
        dummy_labels = np.zeros(len(X_pred_prepared))
        pred_dataset = ExoDataset(X_pred_prepared, dummy_labels)
        pred_loader = DataLoader(pred_dataset, batch_size=config['batch_size'] * 2, shuffle=False)
        
        all_fold_preds = []

        for i, model_path in enumerate(model_files):
            progress(0.3 + (0.6 * (i / len(model_files))), desc=f"Inferring with model {i+1}/{len(model_files)}...")
            
            model = TimeSeriesModel(configs).cuda()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            fold_preds = []
            with torch.no_grad():
                for batch in pred_loader:
                    input_x, _ = batch
                    input_x = input_x.cuda()
                    with autocast("cuda"):
                        output_logits = model(input_x)
                    prediction_probs = torch.sigmoid(output_logits)
                    fold_preds.append(prediction_probs.cpu().numpy())
            
            all_fold_preds.append(np.concatenate(fold_preds))
            del model; gc.collect(); torch.cuda.empty_cache()

        progress(0.9, desc="Averaging predictions...")
        final_predictions = np.mean(np.stack(all_fold_preds, axis=0), axis=0).flatten()

        result_df = pd.DataFrame({
            'Exoplanet_Probability': final_predictions
        })
        result_df['Predicted_Label'] = np.where(result_df['Exoplanet_Probability'] >= 0.5, 'Exoplanet', 'No Exoplanet')
        
        progress(1, desc="Prediction Complete!")
        return result_df, "Prediction successful."

# --- Initial Model Loading ---
PRETRAINED_CONFIGS = [
    {"name": "KOI", "path": "./dataset/cumulative_2025.10.04_09.09.34.csv", "target": "koi_disposition"},
    {"name": "K2", "path": "./dataset/k2pandc_2025.10.04_09.10.31.csv", "target": "disposition"},
    {"name": "TOI", "path": "./dataset/TOI_2025.10.04_09.10.28.csv", "target": "tfopwg_disp"}
]
print("--- Initializing Application ---")
for config in PRETRAINED_CONFIGS:
    name, path, target = config["name"], config["path"], config["target"]
    cache_path = os.path.join(MODEL_CACHE_DIR, f"{name}_model.pkl")
    if os.path.exists(cache_path):
        print(f"Loading '{name}' from cache...")
        with open(cache_path, 'rb') as f: MODEL_DATA[name] = pickle.load(f)
    else:
        print(f"Cache for '{name}' not found. Training new model...")
        prepare_and_train(name, path, target)
print("--- Application Ready ---")

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Exoplanet Hunter") as demo:
    gr.Markdown("# AI-Powered Exoplanet Hunter: Professional Edition")
    gr.Markdown("Using Stacking Ensembles, Bayesian Optimization, SHAP Explainability, and SOTA Time Series Classification.")
    
    model_selector = gr.Dropdown(list(MODEL_DATA.keys()), label="Select Active Dataset (for Ensemble Models)", value="KOI" if "KOI" in MODEL_DATA else None)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Welcome & Instructions", id=0):
            gr.Markdown(
                """
                # Welcome to the AI-Powered Exoplanet Hunter: Professional Edition
                
                This is the complete, local version of the application, giving you full access to explore, train, and evaluate powerful machine learning models for exoplanet candidate classification.

                ## How to Use This Tool: A Practical Guide
                This application is built around two primary workflows: **using pre-trained models** and **creating your own**.

                ---

                ### 🚀 Workflow 1: Using Pre-Trained Models
                *Use this workflow if you have data you want to classify now.*

                #### **Step 1: Explore the Models**
                1.  Navigate to the **Model Details** tab.
                2.  Use the dropdown to select a pre-trained dataset like `KOI`, `K2`, or `TOI`.
                3.  Click **"Show/Refresh Details"** to analyze its performance. You'll see its accuracy, a confusion matrix, ROC curves, and the most important features it learned.

                #### **Step 2: Make Predictions**
                *   **For Tabular Data (Stellar Parameters):**
                    1.  Go to the **Batch Prediction** tab.
                    2.  Choose the pre-trained model you want to use.
                    3.  Upload your CSV file containing the corresponding stellar features.
                    4.  Click **"Run Batch Prediction"** to get the results.

                *   **For Time Series Data (Raw Light Curves):**
                    1.  Go to the **Raw Time Series Classifier** tab.
                    2.  From the dropdown, select a folder of a model you have already trained (e.g., inside the `output_weight` directory).
                    3.  Upload a CSV of raw flux values (see the format instructions in that tab).
                    4.  Click **"Run Prediction"**.

                ---

                ### 🛠️ Workflow 2: Creating and Tuning Your Own Models
                *Use this workflow to build a custom classifier from your own dataset.*

                #### **Step 1: Train a New Model**
                *   **For Tabular Data (Creating an Ensemble):**
                    1.  Go to the **Train Your Own Ensemble** tab.
                    2.  Give your new model a unique name.
                    3.  Upload your CSV dataset.
                    4.  **Crucially, enter the exact name of your dataset's target column** (e.g., `disposition`).
                    5.  Click **"Train Custom Ensemble"**. The new model will become available throughout the app upon completion.

                *   **For Time Series Data (Training a Deep Learning Model):**
                    1.  Go to the **Train Raw Time Series Model** tab.
                    2.  Upload your training CSV, which **must** follow the specified `LABEL, FLUX.1, ...` format.
                    3.  Adjust the hyperparameters (epochs, batch size, learning rate).
                    4.  Click **"Start Training"**. The training logs will appear below, and the final models will be saved in a new folder inside `output_weight`.

                #### **Step 2: Optimize Your Ensemble (Optional)**
                1.  After training a new ensemble model, go to the **Optimize Ensemble** tab.
                2.  Select your newly trained model from the dropdown.
                3.  Choose the number of optimization trials to run.
                4.  Click **"Start Optimization"** to use Bayesian methods (Optuna) to fine-tune your model for potentially better performance. The model details will update automatically.
                
                #### **Step 3: Manage Your Models**
                -   Go to the **Model Management** tab to download any of your trained **ensemble models** as a `.pkl` file for backup or offline use.

                ---

                ### ⚠️ Best Practices & Important Notes
                -   **Data Formatting is Key:** The most common source of errors is incorrectly formatted CSV files. Please read the instructions in each tab carefully before uploading data.
                -   **Computational Cost:** Training new models, especially the time series classifier, is computationally intensive and can take a significant amount of time.
                -   **Required Files:** The time series features require the `models` directory and a `config.json` file to be present in the application's root folder.
                -   **Project Repository:** For a deeper dive into the methodology, code, and advanced usage, please visit the official GitHub repository:
                    **https://github.com/gasthell/NoRush-A-World-Away-Hunting-for-Exoplanets-with-AI**
                """
            )

        with gr.TabItem("Model Details", id=1):
            refresh_button = gr.Button("Show/Refresh Details for Selected Dataset")
            details_markdown = gr.Markdown()
            with gr.Row():
                confusion_matrix_plot = gr.Plot(label="Confusion Matrix")
                roc_curve_plot = gr.Plot(label="ROC Curves")
            shap_summary_plot = gr.Plot(label="Global SHAP Feature Importance")

        with gr.TabItem("Batch Prediction", id=2):
            gr.Markdown("### Predict from a CSV File")
            gr.Markdown("Upload a CSV file containing the same feature columns as the selected model.")
            file_input = gr.File(label="Upload Data CSV")
            file_predict_btn = gr.Button("Run Batch Prediction")
            file_output_df = gr.DataFrame(label="Prediction Results")
            file_status = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("Train Your Own Ensemble", id=3):
            gr.Markdown("### Train a New Stacking Ensemble on Your Data")
            ensemble_name_input = gr.Textbox(label="Enter a Name for Your Ensemble", placeholder="e.g., My TESS Model v1")
            custom_file_upload = gr.File(label="Upload Your CSV Dataset")
            target_col_input = gr.Textbox(label="Target Column", placeholder="e.g., koi_disposition")
            train_button = gr.Button("Train Custom Ensemble")
            train_status_output = gr.Textbox(label="Training Status", interactive=False)

        with gr.TabItem("Optimize Ensemble", id=4):
            gr.Markdown("### Optimize Ensemble with Bayesian Optimization (Optuna)")
            n_trials_slider = gr.Slider(minimum=10, maximum=100, value=20, step=5, label="Number of Optimization Trials")
            optimize_btn = gr.Button("Start Optimization")
            optimization_log = gr.Textbox(label="Optimization Log", lines=15, interactive=False)
        
        with gr.TabItem("Raw Time Series Classifier", id=5, visible=TORCH_AVAILABLE):
            gr.Markdown("## SOTA Time Series Exoplanet Classifier")
            gr.Markdown("This section allows you to use your trained raw time series models to predict on new data.")
            
            with gr.Accordion("Predict on a CSV file", open=True):
                gr.Markdown(
                    """
                    > ### **IMPORTANT: Model and Data Requirements**
                    > 1.  **Model:** The models were trained on time series with a **fixed length of 3198 time steps**.
                    > 2.  **CSV Format:** Your input CSV file **must** contain exactly **3198 columns**, where each column is a flux value at a specific time step, and each row represents a single star's light curve.
                    > -   The file should **not** contain a header row.
                    > -   The file should **not** contain any label or ID columns.
                    
                    **Example of a valid CSV structure (2 stars, 3198 flux values each):**
                    ```                    -10.1, -12.5, ..., 5.4  <-- Row 1: 3198 values for Star 1
                    8.2, 9.1, ..., -2.7   <-- Row 2: 3198 values for Star 2
                    ```
                    """
                )
                ts_model_folder_selector = gr.Dropdown(choices=get_model_folders(), label="Select Trained Model Folder (from output_weight)", interactive=True)
                ts_prediction_file = gr.File(label="Upload CSV for Prediction")
                ts_predict_button = gr.Button("Run Prediction", variant="primary")
                
                ts_prediction_output = gr.DataFrame(label="Prediction Results")
                ts_prediction_status = gr.Textbox(label="Status", interactive=False)
                

        with gr.TabItem("Train Raw Time Series Model", id=6, visible=TORCH_AVAILABLE):
            gr.Markdown("## Train the SOTA Time Series Classifier")
            gr.Markdown("Upload your dataset, configure the hyperparameters, and start the training process. The model will be trained using 5-fold cross-validation.")
            
            gr.Markdown(
                """
                > ### **Important: Dataset Format**
                > For the training to succeed, your uploaded CSV file **must** adhere to the following format:
                > - The first column must be named `LABEL`.
                > - The `LABEL` column must contain the ground truth, where `2` indicates a confirmed exoplanet and `1` indicates no exoplanet.
                > - All subsequent columns (`FLUX.1`, `FLUX.2`, etc.) must be the numerical flux values that constitute the time series data.
                > - There should be no missing values in the flux data.
                """
            )

            ts_dataset_upload = gr.File(label="Upload Time Series CSV Dataset", type="filepath")

            with gr.Row():
                ts_epochs = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Epochs")
                ts_batch_size = gr.Slider(minimum=8, maximum=128, value=32, step=4, label="Batch Size")
                ts_lr = gr.Number(value=1e-4, label="Learning Rate", info="Initial learning rate for AdamW optimizer.")
            
            start_ts_training_btn = gr.Button("Start Training", variant="primary")
            ts_training_log = gr.Textbox(label="Training Log", lines=20, interactive=False, placeholder="Training logs will be displayed here...")

        with gr.TabItem("Model Management", id=7):
            gr.Markdown("### Download a Trained Model")
            download_button = gr.Button("Download Selected Model")
            download_file_output = gr.File(label="Download Link")
    
    # --- Event Wiring ---
    details_outputs = [details_markdown, confusion_matrix_plot, roc_curve_plot, shap_summary_plot]
    refresh_button.click(fn=get_full_model_details, inputs=model_selector, outputs=details_outputs)
    model_selector.change(fn=get_full_model_details, inputs=model_selector, outputs=details_outputs)
    
    file_predict_btn.click(fn=predict_from_file, inputs=[model_selector, file_input], outputs=[file_output_df, file_status])

    train_button.click(fn=prepare_and_train, inputs=[ensemble_name_input, custom_file_upload, target_col_input], outputs=[train_status_output, model_selector])

    optimize_outputs = [optimization_log, details_markdown, confusion_matrix_plot, roc_curve_plot, shap_summary_plot]
    optimize_btn.click(fn=bayesian_optimizer, inputs=[model_selector, n_trials_slider], outputs=optimize_outputs)

    download_button.click(fn=download_model, inputs=model_selector, outputs=download_file_output)
    
    if TORCH_AVAILABLE:
        start_ts_training_btn.click(fn=start_ts_training, inputs=[ts_dataset_upload, ts_epochs, ts_batch_size, ts_lr], outputs=ts_training_log)
        ts_predict_button.click(fn=run_ts_prediction, inputs=[ts_model_folder_selector, ts_prediction_file], outputs=[ts_prediction_output, ts_prediction_status])

    
    demo.load(fn=get_full_model_details, inputs=model_selector, outputs=details_outputs)

demo.launch(debug=True)