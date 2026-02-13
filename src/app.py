import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import create_app

app = create_app()

if __name__ == '__main__':
    # Run dev server
    # Host 0.0.0.0 allows access from other devices if needed
    print("Starting Retention System v2...")
    print("Access at http://localhost:5000")
    app.run(debug=True, port=5000)
