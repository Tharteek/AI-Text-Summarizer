import os
import subprocess
import sys

def main():
    print("Welcome to GAN-T5 Text Summarizer Project!")
    print("------------------------------------------")
    
    # 1. Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("\n[1] Project Structure:")
    print(" - src/         : Core implementation (Generator, Discriminator, Train)")
    print(" - models/      : Saved model weights")
    print(" - docs/        : Detailed analysis report")
    print(" - app.py       : Streamlit Web Dashboard")
    
    print("\n[2] How to proceed?")
    print(" - To train the model: run 'python src/train.py'")
    print(" - To see the analysis: read 'docs/analysis.md'")
    print(" - To launch the UI: run 'streamlit run app.py'")
    
    print("\nStarting the Streamlit dashboard as a demonstration...")
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
