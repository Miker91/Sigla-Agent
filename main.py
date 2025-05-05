import os
import subprocess
import sys

def main():
    print("Starting Sigla Pandas Data Analyst App...")
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the app
    app_path = os.path.join(current_dir, "apps", "pandas-data-analyst-app", "app.py")
    
    # Check if the app exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find app at {app_path}")
        return
    
    try:
        # Run the streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except Exception as e:
        print(f"Error running the app: {e}")


if __name__ == "__main__":
    main()
