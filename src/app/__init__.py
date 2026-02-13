from flask import Flask, render_template
from flask_cors import CORS

def create_app():
    # Initialize Flask
    # Template folder is ../templates relative to where this file is (src/app)
    # Actually, if we run from root (python src/app.py), templates should be in src/templates?
    # Standard Flask: 'templates' folder relative to app root.
    # Let's set it explicitly to src/templates for clarity.
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Enable CORS
    CORS(app)
    
    # Register API Blueprint
    from src.app.routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Serve Frontend (Single Page App)
    @app.route('/')
    def index():
        return render_template('index.html')
        
    return app
