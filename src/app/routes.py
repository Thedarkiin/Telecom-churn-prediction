import os
import joblib
import pandas as pd
from flask import Blueprint, request, jsonify, make_response
from src.app.utils import preprocess_input, preprocess_batch

PREDICTION_HISTORY = [] # In-memory session storage

bp = Blueprint('api', __name__)

class ModelLoader:
    _instance = None
    _model = None
    _feature_names = None
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                cls._model = joblib.load('models/xgboost_model.pkl')
                cls._feature_names = joblib.load('models/feature_names.pkl')
                try:
                    cls._scaler = joblib.load('models/scaler.pkl')
                    cls._imputer = joblib.load('models/numeric_imputer.pkl')
                except:
                    print("Warning: Scaler/Imputer not found. Inference may be inaccurate.")
                    cls._scaler = None
                    cls._imputer = None
            except FileNotFoundError:
                print("Model files not found. Please run the pipeline first.")
                return None, None
        return cls._model, cls._feature_names, cls._scaler, cls._imputer

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
        model, feature_names, scaler, imputer = ModelLoader.get_model()
        if model is None:
            return jsonify({'error': 'Model not trained (artifacts missing)'}), 503
            
        # Preprocess
        processed_df = preprocess_input(data, feature_names, scaler, imputer)
        
        # Predict
        prob = model.predict_proba(processed_df)[0][1]
        
        # Load Causal Insights
        insights = []
        try:
            import json
            with open('models/double_ml_summary.json', 'r') as f:
                causal_data = json.load(f)
            

            
            # Re-implement simple lookup
            if f"Contract_{data.get('Contract')}" in causal_data:
                item = causal_data[f"Contract_{data.get('Contract')}"]
                insights.append({'description': item['description'], 'action': item['action']})
                
            if f"InternetService_{data.get('InternetService')}" in causal_data:
                item = causal_data[f"InternetService_{data.get('InternetService')}"]
                insights.append({'description': item['description'], 'action': item['action']})

            if f"PaymentMethod_{data.get('PaymentMethod')}" in causal_data:
                item = causal_data[f"PaymentMethod_{data.get('PaymentMethod')}"]
                insights.append({'description': item['description'], 'action': item['action']})

        except Exception as e:
            print(f"Error loading causal insights: {e}")
            
        result = {
            'churn_probability': float(prob),
            'risk_level': 'High' if prob > 0.3 else 'Low',
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
            
            model, feature_names, scaler, imputer = ModelLoader.get_model()
            if model is None:
                return jsonify({'error': 'Model not trained'}), 503
                
            processed_df = preprocess_batch(df, feature_names, scaler, imputer)
            probs = model.predict_proba(processed_df)[:, 1]
            
            results = []
            for i, prob in enumerate(probs):
                record = df.iloc[i].to_dict()
                clean_record = {k: (v if pd.notna(v) else None) for k, v in record.items()}
                
                res = {
                    'churn_probability': float(prob),
                    'risk_level': 'High' if prob > 0.3 else 'Low',
                    'input_data': clean_record
                }
                results.append(res)
                
            # Add batch to history (reverse so first item in CSV is at top of recent history?)
            # Actually insert(0) puts it at top.
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
