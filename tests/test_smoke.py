import pytest
import pandas as pd
import os
from src.pipeline import main
from src.config import Config

def test_pipeline_smoke():
    """
    Smoke test to ensure the pipeline runs end-to-end on a small subset of data.
    This doesn't check model quality, just that the code doesn't crash.
    """
    # 1. Setup: Create a temporary config or modify the existing one to be fast
    # For simplicity, we'll rely on the main pipeline but we could mock data.
    # A better approach for a smoke test is to use the actual data but maybe restrict iterations.
    
    # Let's force a "smoke mode" by monkeypatching the Config if needed, 
    # or just run it and ensure it passes.
    # Since the dataset is small (~3k rows), running the full pipeline might be fast enough (< 1 min).
    # If it's too slow, we should create a 'smoke_test' flag in the pipeline.
    
    # For now, let's try running the main function.
    try:
        # We can also start with a smaller dataset if we want to be safe, 
        # but let's test the 'real' pipeline first.
        models, metrics = main()
        
        # 2. Assertions
        assert models is not None, "Models dictionary should not be None"
        assert len(models) > 0, "Should have trained at least one model"
        assert not metrics.empty, "Metrics dataframe should not be empty"
        assert 'git_hash' in metrics.columns, "Metrics should contain git_hash"
        
        # Check artifacts exist
        assert os.path.exists(Config.METRICS_PATH), "Metrics directory should exist"
        assert os.path.exists(os.path.join(Config.METRICS_PATH, "all_metrics.csv")), "Metrics CSV should exist"
        
    except Exception as e:
        pytest.fail(f"Pipeline crashed: {e}")
