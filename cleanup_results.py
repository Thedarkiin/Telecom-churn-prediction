"""
Cleanup script to remove unnecessary result files.
Keeps only essential metrics and visualizations.
"""

import os
import glob

def cleanup_results():
    """Remove unnecessary LIME files and other redundant results."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    explainability_dir = os.path.join(base_dir, "results", "explainability")
    
    # Files to delete
    lime_html_files = glob.glob(os.path.join(explainability_dir, "lime_explanation_*.html"))
    lime_csv_files = glob.glob(os.path.join(explainability_dir, "lime_weights_*.csv"))
    
    files_to_delete = lime_html_files + lime_csv_files
    
    print(f"Found {len(files_to_delete)} files to delete:")
    print(f"  - {len(lime_html_files)} LIME HTML explanation files")
    print(f"  - {len(lime_csv_files)} LIME weight CSV files")
    
    if len(files_to_delete) == 0:
        print("No files to delete. Directory already clean!")
        return
    
    # Confirm deletion
    response = input("\nProceed with deletion? (y/n): ")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Delete files
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"\n✓ Deleted {deleted_count} files")
    print(f"✓ Kept essential files:")
    
    kept_files = glob.glob(os.path.join(explainability_dir, "*"))
    kept_files = [os.path.basename(f) for f in kept_files if os.path.isfile(f)]
    for f in kept_files:
        print(f"  - {f}")

if __name__ == "__main__":
    cleanup_results()
