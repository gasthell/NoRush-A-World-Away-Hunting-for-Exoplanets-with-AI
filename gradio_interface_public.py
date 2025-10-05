import gradio as gr
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import glob

# ML Imports
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import KNNImputer
from itertools import cycle

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import GradScaler, autocast
    # This requires the 'models' directory from your notebook to be in the same folder
    from models import TimerBackbone
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch or the 'models.TimerBackbone' module is not available. The Time Series Classifier tab will be disabled.")

import json
import gc

# --- Global Settings ---
MODEL_DATA = {}
MODEL_CACHE_DIR = "saved_models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# --- Gradio UI Functions ---
def get_full_model_details(dataset_name):
    plt.close('all')
    if dataset_name not in MODEL_DATA: return "No model selected or model data is missing.", None, None, None

    data = MODEL_DATA[dataset_name]
    # Check for essential keys to prevent errors if model loading failed
    if not all(k in data for k in ['y_test', 'X_test', 'features', 'ensemble_model', 'class_names']):
        return "Model data is incomplete. Cannot generate details.", None, None, None

    y_test = data['y_test']
    X_test_df = pd.DataFrame(data['X_test'], columns=data['features'])
    y_pred = data['ensemble_model'].predict(X_test_df)
    y_prob = data['ensemble_model'].predict_proba(X_test_df)
    class_names = data['class_names']
    
    features_md = "### Features Used (" + str(len(data['features'])) + " total)\n" + ", ".join(f"`{f}`" for f in data['features'])
    report = f"# Report: {dataset_name}\n**Ensemble Accuracy**: {data['accuracy']:.4f}\n\n" + features_md
    
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6)); 
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=class_names); 
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues); 
    ax_cm.set_title("Confusion Matrix"); 
    plt.tight_layout()
    
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
        if len(class_names) == 2: # SHAP summary for binary case needs specific handling
            shap.summary_plot(data['shap_values'][1], data['X_sample'], plot_type="bar", show=False)
        else: # Multi-class
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
            # Load without a header
            df_pred = pd.read_csv(prediction_file.name, header=None)
            
            # --- Data Validation ---
            expected_cols = 3197
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
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for i, model_path in enumerate(model_files):
            progress(0.3 + (0.6 * (i / len(model_files))), desc=f"Inferring with model {i+1}/{len(model_files)}...")
            
            model = TimeSeriesModel(configs).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            fold_preds = []
            with torch.no_grad():
                for batch in pred_loader:
                    input_x, _ = batch
                    input_x = input_x.to(device)
                    with autocast(device_type=device):
                        output_logits = model(input_x)
                    prediction_probs = torch.sigmoid(output_logits)
                    fold_preds.append(prediction_probs.cpu().numpy())
            
            all_fold_preds.append(np.concatenate(fold_preds))
            del model; gc.collect(); 
            if device == "cuda":
                torch.cuda.empty_cache()

        progress(0.9, desc="Averaging predictions...")
        final_predictions = np.mean(np.stack(all_fold_preds, axis=0), axis=0).flatten()

        result_df = pd.DataFrame({
            'Exoplanet_Probability': final_predictions
        })
        result_df['Predicted_Label'] = np.where(result_df['Exoplanet_Probability'] >= 0.5, 'Exoplanet', 'No Exoplanet')
        
        progress(1, desc="Prediction Complete!")
        return result_df, "Prediction successful."

# --- Initial Model Loading ---
# This application relies on pre-trained models being available in the `saved_models` directory.
PRETRAINED_MODELS = ["KOI", "K2", "TOI"]
print("--- Initializing Application ---")
for name in PRETRAINED_MODELS:
    cache_path = os.path.join(MODEL_CACHE_DIR, f"{name}_model.pkl")
    if os.path.exists(cache_path):
        print(f"Loading '{name}' from cache...")
        try:
            with open(cache_path, 'rb') as f: 
                MODEL_DATA[name] = pickle.load(f)
        except Exception as e:
            print(f"Error loading model '{name}' from cache: {e}")
    else:
        print(f"WARNING: Cache for '{name}' not found at {cache_path}. This model will be unavailable.")
print("--- Application Ready ---")

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Exoplanet Hunter") as demo:
    gr.Markdown("# AI-Powered Exoplanet Hunter")
    gr.Markdown("An interface for predicting exoplanet candidates using pre-trained machine learning models.")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Welcome & Instructions", id=0):
            gr.Markdown(
                """
                # Welcome to the AI-Powered Exoplanet Hunter
                
                This interactive tool demonstrates how pre-trained machine learning models can be used to classify exoplanet candidates from real NASA datasets. Welcome!

                ## How to Use This Demonstration
                This application is designed for two main activities: exploring the performance of our pre-trained models and using them to make predictions on your own data.

                ---

                ### 🚀 **Part 1: Explore the Pre-Trained Models**
                
                Get a feel for how the AI performs before you use it.

                1.  Navigate to the **Model Details** tab.
                2.  Use the dropdown menu to select a dataset like `KOI`, `K2`, or `TOI`.
                3.  Click the **"Show/Refresh Details"** button.
                4.  You can now analyze the model's historical performance, including its overall accuracy, a confusion matrix (to see where it makes mistakes), ROC curves, and a SHAP plot showing which stellar features were most important for its decisions.

                ---

                ### 🛠️ **Part 2: Predict on Your Own Data**
                
                Once you've seen how the models work, use them to classify new potential exoplanets.

                #### **If you have Tabular Data (Stellar Parameters):**
                1.  Go to the **Ensemble Model Prediction** tab.
                2.  Choose the pre-trained model (`KOI`, `K2`, or `TOI`) that matches your data's feature set.
                3.  Upload your CSV file containing the stellar parameters.
                4.  Click **"Run Batch Prediction"** to receive the classification for each candidate in your file.

                #### **If you have Time Series Data (Raw Light Curves):**
                1.  Go to the **Raw Time Series Classifier** tab.
                2.  From the dropdown, select one of the provided, pre-trained model folders.
                3.  Upload your CSV file containing only the raw flux values. **This file must be formatted correctly to work.** Please follow the detailed instructions and example provided in that tab.
                4.  Click **"Run Prediction"** to get a probability score for each light curve.

                ---

                ### 📁 **Model Management**

                -   Interested in using the models offline? Go to the **Model Management** tab to download any of the pre-trained **ensemble models** as a `.pkl` file.

                ---

                ### ⚠️ **About This Public Version**

                -   **Demonstration Only:** This is a public-facing demonstration. Features for training new models and running hyperparameter optimization have been disabled.
                -   **Get the Full Version:** To access the complete tool with all training and optimization capabilities, please visit our project repository on GitHub. The full code and instructions are available there.
                
                **https://github.com/gasthell/NoRush-A-World-Away-Hunting-for-Exoplanets-with-AI**
                """
            )
            
        with gr.TabItem("Model Details", id=1):
            model_selector_details = gr.Dropdown(list(MODEL_DATA.keys()), label="Select Dataset to View Details", value="KOI" if "KOI" in MODEL_DATA else None)
            refresh_button = gr.Button("Show/Refresh Details for Selected Dataset")
            details_markdown = gr.Markdown()
            with gr.Row():
                confusion_matrix_plot = gr.Plot(label="Confusion Matrix")
                roc_curve_plot = gr.Plot(label="ROC Curves")
            shap_summary_plot = gr.Plot(label="Global SHAP Feature Importance")

        with gr.TabItem("Ensemble Model Prediction", id=2):
            gr.Markdown("### Predict from a CSV File using an Ensemble Model")
            gr.Markdown("Upload a CSV file containing stellar parameter columns matching the selected model.")
            model_selector_predict = gr.Dropdown(list(MODEL_DATA.keys()), label="Select Active Ensemble Model", value="KOI" if "KOI" in MODEL_DATA else None)
            file_input = gr.File(label="Upload Data CSV")
            file_predict_btn = gr.Button("Run Batch Prediction")
            file_output_df = gr.DataFrame(label="Prediction Results")
            file_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.TabItem("Raw Time Series Classifier", id=3, visible=TORCH_AVAILABLE):
            gr.Markdown("## Raw Time Series Exoplanet Classifier")
            gr.Markdown("This section allows you to use your trained raw time series models to predict on new data.")
            
            with gr.Accordion("Predict on a CSV file", open=True):
                gr.Markdown(
                    """
                    > ### **IMPORTANT: Model and Data Requirements**
                    > 1.  **Model:** The models were trained on time series with a **fixed length of 3197 time steps**.
                    > 2.  **CSV Format:** Your input CSV file **must** contain exactly **3197 columns**, where each column is a flux value at a specific time step, and each row represents a single star's light curve.
                    > -   The file should **not** contain a header row.
                    > -   The file should **not** contain any label or ID columns.
                    
                    **Example of a valid CSV structure (2 stars, 3197 flux values each):**
                    ```                    -10.1,-12.5,...,5.4
                    8.2,9.1,...,-2.7
                    ```
                    """
                )
                ts_model_folder_selector = gr.Dropdown(choices=get_model_folders(), label="Select Trained Model Folder (from output_weight)", interactive=True)
                ts_prediction_file = gr.File(label="Upload CSV for Prediction")
                ts_predict_button = gr.Button("Run Prediction", variant="primary")
                
                ts_prediction_output = gr.DataFrame(label="Prediction Results")
                ts_prediction_status = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("Model Management", id=4):
            gr.Markdown("### Download a Pre-Trained Ensemble Model")
            model_selector_download = gr.Dropdown(list(MODEL_DATA.keys()), label="Select Model to Download", value="KOI" if "KOI" in MODEL_DATA else None)
            download_button = gr.Button("Download Selected Model")
            download_file_output = gr.File(label="Download Link", interactive=False)
    
    # --- Event Wiring ---
    details_outputs = [details_markdown, confusion_matrix_plot, roc_curve_plot, shap_summary_plot]
    refresh_button.click(fn=get_full_model_details, inputs=model_selector_details, outputs=details_outputs)
    model_selector_details.change(fn=get_full_model_details, inputs=model_selector_details, outputs=details_outputs)
    
    file_predict_btn.click(fn=predict_from_file, inputs=[model_selector_predict, file_input], outputs=[file_output_df, file_status])

    download_button.click(fn=download_model, inputs=model_selector_download, outputs=download_file_output)
    
    if TORCH_AVAILABLE:
        ts_predict_button.click(fn=run_ts_prediction, inputs=[ts_model_folder_selector, ts_prediction_file], outputs=[ts_prediction_output, ts_prediction_status])

    demo.load(fn=get_full_model_details, inputs=model_selector_details, outputs=details_outputs)

# To make it publicly accessible, use share=True
demo.launch(share=True)