import subprocess
import pytest
import os
import sys

def test_pipeline_script_execution():
    """
    Smoke test that runs 'pipeline.py' as a subprocess.
    This verifies the script runs from start to finish without crashing.
    It avoids module import issues by treating the pipeline as a black box.
    """
    # Get the path to pipeline.py (in the root directory)
    # We assume tests are run from the root, but let's be robust
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    pipeline_path = os.path.join(root_dir, "pipeline.py")

    # Command to run: python pipeline.py
    # We capture stdout/stderr to inspect if it fails
    # We can add a timeout to prevent it from hanging forever (e.g., 300 seconds)
    try:
        result = subprocess.run(
            [sys.executable, pipeline_path],
            cwd=root_dir,  # Run from root so relative paths (data/, src/) work
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check return code (0 means success)
        if result.returncode != 0:
            # Failed! Print stderr for debugging
            pytest.fail(f"Pipeline script failed!\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        
        # Additional checks: Verify artifacts were created
        metrics_csv = os.path.join(root_dir, "results", "metrics", "all_metrics.csv")
        assert os.path.exists(metrics_csv), "all_metrics.csv was not created!"
        
    except subprocess.TimeoutExpired:
        pytest.fail("Pipeline execution timed out after 5 minutes!")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")
