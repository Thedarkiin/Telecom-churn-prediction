
import os
import joblib
import pandas as pd
from flask import Blueprint, request, jsonify, make_response
from src.preprocessing import ChurnPreprocessor

PREDICTION_HISTORY = [] # In-memory session storage

bp = Blueprint('api', __name__)

class ModelLoader:
    _instance = None
    _model = None
    _preprocessor = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
            try:
                # Load Unified Artifacts
                if os.path.exists('models/ensemble_model.pkl'):
                    cls._model = joblib.load('models/ensemble_model.pkl')
                    print("Loaded Ensemble Model (Best Balanced)")
                elif os.path.exists('models/xgboost_model.pkl'):
                    cls._model = joblib.load('models/xgboost_model.pkl')
                    print("Loaded XGBoost Model")
                
                if os.path.exists('models/preprocessor.pkl'):
                    cls._preprocessor = joblib.load('models/preprocessor.pkl') 
                print("Models loaded successfully")
            except Exception as e:
                print(f"Error loading models: {e}")
                cls._model = None
                cls._preprocessor = None
        return cls._instance

    def predict(self, data_dict):
        if self._model is None or self._preprocessor is None:
            return None, "Model not loaded", None, []
            
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([data_dict])
            
            # TRANSFORM using the Unified Preprocessor
            processed_df = self._preprocessor.transform(input_df)
            
            # Predict
            prob = self._model.predict_proba(processed_df)[0][1]
            
            # Get Risk Level
            risk = "High" if prob > 0.5 else "Low" 
            
            # Get Insights (from pre-computed JSON)
            insights = []
            try:
                import json
                if os.path.exists('models/double_ml_summary.json'):
                    with open('models/double_ml_summary.json', 'r') as f:
                         dml_summary = json.load(f)
                    
                    # Check active features against recommendations
                    for key, impact in dml_summary.items():
                        # 1. Direct match (e.g. key=Contract_Two year, input: Contract="Two year")
                        is_match = False
                        if "_" in key:
                             feat, val = key.split("_", 1)
                             # Check raw input data
                             if feat in data_dict and str(data_dict[feat]) == val:
                                 is_match = True
                        
                        # 2. Boolean match if not split (e.g. "SeniorCitizen" -> 1)
                        elif key in data_dict and str(data_dict[key]) == '1':
                             is_match = True
                        
                        if is_match:
                             insights.append(impact)
                             
            except Exception as e:
                print(f"Insight error: {e}")
                
            return prob, risk, processed_df, insights
            
        except Exception as e:
            return None, str(e), None, []

@bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Load Model
        model_loader = ModelLoader.get_model()
        if model_loader._model is None or model_loader._preprocessor is None:
            return jsonify({'error': 'Model not trained (artifacts missing)'}), 503
            
        prob, risk, _, insights = model_loader.predict(data)
        
        if prob is None:
             return jsonify({'error': risk}), 500
            
        result = {
            'churn_probability': float(prob),
            'risk_level': risk,
            'causal_insights': insights,
            'input_data': data
        }
        
        PREDICTION_HISTORY.insert(0, result)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/history', methods=['GET'])
def get_history():
    return jsonify(PREDICTION_HISTORY)

@bp.route('/history', methods=['DELETE'])
def clear_history():
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    return jsonify({'status': 'cleared'})



@bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file:
            df = pd.read_csv(file)
            
            model_loader = ModelLoader.get_model()
            if model_loader._model is None:
                return jsonify({'error': 'Model not trained'}), 503
                
            results = []
            records = df.to_dict(orient='records')
            
            for record in records:
                prob, risk, _, _ = model_loader.predict(record)
                if prob is not None:
                    clean_record = {k: (v if pd.notna(v) else None) for k, v in record.items()}
                    res = {
                        'churn_probability': float(prob),
                        'risk_level': risk,
                        'input_data': clean_record
                    }
                    results.append(res)
                
            for r in results:
                PREDICTION_HISTORY.insert(0, r)
                
            return jsonify({'message': f'Processed {len(results)} records', 'results': results})
            
    except Exception as e:
         return jsonify({'error': str(e)}), 500

@bp.route('/export', methods=['GET'])
def export_history():
    try:
        if not PREDICTION_HISTORY:
            return jsonify({'error': 'No history to export'}), 400
            
        flat_data = []
        for item in PREDICTION_HISTORY:
            row = item['input_data'].copy()
            row['Churn_Probability'] = item['churn_probability']
            row['Risk_Level'] = item['risk_level']
            flat_data.append(row)
            
        df = pd.DataFrame(flat_data)
        
        csv_data = df.to_csv(index=False)
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = 'attachment; filename=churn_predictions.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/active_template', methods=['GET'])
def get_template():
    # Create a simple template dataframe
    data = {
        'tenure': [12, 24, 1],
        'MonthlyCharges': [70.5, 89.1, 29.9],
        'TotalCharges': [846.0, 2138.4, 29.9],
        'Contract': ['One year', 'Two year', 'Month-to-month'],
        'PaymentMethod': ['Mailed check', 'Credit card (automatic)', 'Electronic check'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'SeniorCitizen': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)
    
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = 'attachment; filename=churn_template.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response
