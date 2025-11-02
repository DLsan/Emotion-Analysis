# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# Dataset (too large for Docker image)
FER-2013/

# Training outputs (optional - include only final model)
training_history.pkl
training_history.png
plots/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Logs
*.log

# Screenshots
screenshots/

# Other models (only include the best one)
emotion_model.h5
emotion_model_improved.h5
# Keep only: best_emotion_model_80.h5

# Misc
download_model.py
prepare_fer2013_dataset.py
check_dataset_structure.py
diagnose_dataset.py
README.md
LICENSE