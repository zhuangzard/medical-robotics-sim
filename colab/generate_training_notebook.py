#!/usr/bin/env python3
"""
Generate Colab training notebook for medical-robotics-sim
Generates notebook in project's notebooks/ directory
"""

import json
from pathlib import Path
from datetime import datetime

def generate_notebook():
    """Generate complete Colab training notebook"""
    
    cells = []
    
    # Cell 1: Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Medical Robotics - Week 1 Training\n",
            "\n",
            "**Project**: Physics-Informed Foundation Models for Medical Robotics  \n",
            "**Goal**: Train PPO, GNS, and PhysRobot on PushBox task  \n",
            "**Target**: 12.5x sample efficiency, >95% OOD generalization  \n",
            "\n",
            "**GitHub**: https://github.com/zhuangzard/medical-robotics-sim  \n",
            "**Generated**: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  \n",
            "\n",
            "---\n",
            "\n",
            "## ⚠️ Setup Required\n",
            "\n",
            "1. **Runtime**: Change to GPU (Runtime → Change runtime type)\n",
            "2. **GPU Type**: Select V100 or A100 (Colab Pro)\n",
            "3. **Run All**: Runtime → Run all\n",
            "\n",
            "---"
        ]
    })
    
    # Cell 2: GPU Detection
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 🔍 GPU Detection\n",
            "import subprocess\n",
            "import torch\n",
            "\n",
            "print('='*60)\n",
            "print('🎮 GPU Configuration')\n",
            "print('='*60)\n",
            "\n",
            "# Check GPU\n",
            "try:\n",
            "    gpu_info = subprocess.check_output(\n",
            "        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']\n",
            "    ).decode('utf-8').strip()\n",
            "    print(f'GPU: {gpu_info}')\n",
            "except:\n",
            "    print('❌ No GPU detected! Please change runtime to GPU.')\n",
            "\n",
            "# PyTorch check\n",
            "print(f'PyTorch: {torch.__version__}')\n",
            "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    gpu_name = torch.cuda.get_device_name(0)\n",
            "    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
            "    print(f'GPU Memory: {gpu_mem:.1f} GB')\n",
            "    \n",
            "    # Auto-configure\n",
            "    if 'A100' in gpu_name:\n",
            "        batch_size, workers = 128, 8\n",
            "        print('🚀 A100 detected: batch_size=128, workers=8')\n",
            "    elif 'V100' in gpu_name:\n",
            "        batch_size, workers = 64, 4\n",
            "        print('🚀 V100 detected: batch_size=64, workers=4')\n",
            "    else:\n",
            "        batch_size, workers = 32, 2\n",
            "        print('🚀 T4 detected: batch_size=32, workers=2')\n",
            "else:\n",
            "    batch_size, workers = 16, 2\n",
            "    print('⚠️  CPU mode (slow!)')\n",
            "\n",
            "print('='*60)"
        ]
    })
    
    # Cell 3: Install Dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📦 Install Dependencies\n",
            "print('Installing dependencies...')\n",
            "\n",
            "!pip install -q torch torchvision torchaudio\n",
            "!pip install -q torch-geometric\n",
            "!pip install -q gymnasium mujoco\n",
            "!pip install -q stable-baselines3\n",
            "!pip install -q matplotlib numpy scipy tqdm\n",
            "\n",
            "print('✅ Dependencies installed!')"
        ]
    })
    
    # Cell 4: Clone Repository
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📥 Clone Project Repository\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "REPO_URL = 'https://github.com/zhuangzard/medical-robotics-sim'\n",
            "REPO_NAME = 'medical-robotics-sim'\n",
            "\n",
            "if not Path(REPO_NAME).exists():\n",
            "    print(f'Cloning {REPO_NAME}...')\n",
            "    !git clone {REPO_URL}\n",
            "    print('✅ Repository cloned')\n",
            "else:\n",
            "    print(f'{REPO_NAME} exists, pulling latest...')\n",
            "    %cd {REPO_NAME}\n",
            "    !git pull\n",
            "    %cd ..\n",
            "\n",
            "%cd {REPO_NAME}\n",
            "print(f'\\n📂 Working directory: {os.getcwd()}')\n",
            "!ls -la"
        ]
    })
    
    # Cell 5: Setup Progress Tracking
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📊 Setup Progress Tracking\n",
            "from google.colab import drive\n",
            "import json\n",
            "from datetime import datetime\n",
            "from pathlib import Path\n",
            "\n",
            "# Mount Drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Create progress directory\n",
            "progress_dir = Path('/content/drive/MyDrive/medical-robotics-progress')\n",
            "progress_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "progress_file = progress_dir / 'training_progress.json'\n",
            "\n",
            "def update_progress(status, **kwargs):\n",
            "    \"\"\"Update progress file in Drive\"\"\"\n",
            "    progress = {\n",
            "        'timestamp': datetime.now().isoformat(),\n",
            "        'status': status,\n",
            "        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',\n",
            "        **kwargs\n",
            "    }\n",
            "    with open(progress_file, 'w') as f:\n",
            "        json.dump(progress, f, indent=2)\n",
            "    print(f'📊 Progress updated: {status}')\n",
            "\n",
            "update_progress('started', message='Training initialization complete')\n",
            "\n",
            "print(f'✅ Progress tracking setup')\n",
            "print(f'📁 Progress file: {progress_file}')"
        ]
    })
    
    # Cell 6: Training
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 🚀 Start Training\n",
            "import time\n",
            "\n",
            "print('='*60)\n",
            "print(f'🏁 Training Started: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')\n",
            "print('='*60)\n",
            "\n",
            "start_time = time.time()\n",
            "\n",
            "try:\n",
            "    # Update progress\n",
            "    update_progress('training', message='Training in progress', eta_hours=8)\n",
            "    \n",
            "    # Run training script\n",
            "    !bash experiments/week1_push_box/setup_and_run.sh\n",
            "    \n",
            "    # Training complete\n",
            "    duration_sec = time.time() - start_time\n",
            "    duration_hr = duration_sec / 3600\n",
            "    \n",
            "    update_progress('complete', \n",
            "                   message='Training complete', \n",
            "                   duration_hours=duration_hr)\n",
            "    \n",
            "    print('='*60)\n",
            "    print('✅ Training Complete!')\n",
            "    print(f'⏱️  Duration: {duration_hr:.1f} hours')\n",
            "    print('='*60)\n",
            "    \n",
            "except Exception as e:\n",
            "    update_progress('error', message=str(e))\n",
            "    print(f'❌ Error: {e}')\n",
            "    raise"
        ]
    })
    
    # Cell 7: Save Results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 💾 Save Results to Drive\n",
            "import shutil\n",
            "\n",
            "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
            "results_dir = Path(f'/content/drive/MyDrive/medical-robotics-results/{timestamp}')\n",
            "results_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f'📁 Saving results to: {results_dir}')\n",
            "\n",
            "# Copy directories\n",
            "dirs_to_save = ['results', 'models', 'data']\n",
            "for dir_name in dirs_to_save:\n",
            "    src = Path(dir_name)\n",
            "    if src.exists():\n",
            "        dst = results_dir / dir_name\n",
            "        shutil.copytree(src, dst, dirs_exist_ok=True)\n",
            "        print(f'  ✅ {dir_name}/')\n",
            "\n",
            "# Copy key files\n",
            "files_to_save = ['*.json', '*.md', '*.png', '*.pdf', '*.tex']\n",
            "saved_count = 0\n",
            "for pattern in files_to_save:\n",
            "    for file in Path('.').glob(pattern):\n",
            "        if file.is_file():\n",
            "            shutil.copy2(file, results_dir / file.name)\n",
            "            saved_count += 1\n",
            "\n",
            "print(f'  ✅ {saved_count} files')\n",
            "\n",
            "# Create summary\n",
            "summary = {\n",
            "    'timestamp': timestamp,\n",
            "    'duration_hours': duration_hr,\n",
            "    'gpu': torch.cuda.get_device_name(0),\n",
            "    'status': 'complete',\n",
            "    'drive_path': str(results_dir)\n",
            "}\n",
            "\n",
            "with open(results_dir / 'summary.json', 'w') as f:\n",
            "    json.dump(summary, f, indent=2)\n",
            "\n",
            "update_progress('saved', message='Results saved to Drive', path=str(results_dir))\n",
            "\n",
            "print(f'\\n✅ All results saved to Drive!')\n",
            "print(f'📂 {results_dir}')"
        ]
    })
    
    # Cell 8: Display Results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📊 Display Results\n",
            "from IPython.display import Image, Markdown, display\n",
            "\n",
            "print('='*60)\n",
            "print('📊 Week 1 Training Results')\n",
            "print('='*60)\n",
            "\n",
            "# Display Table 1\n",
            "table1_path = 'results/tables/sample_efficiency.md'\n",
            "if Path(table1_path).exists():\n",
            "    print('\\n📋 Table 1: Sample Efficiency Comparison\\n')\n",
            "    with open(table1_path) as f:\n",
            "        display(Markdown(f.read()))\n",
            "\n",
            "# Display Figure 2\n",
            "fig2_path = 'results/figures/ood_generalization.png'\n",
            "if Path(fig2_path).exists():\n",
            "    print('\\n📈 Figure 2: OOD Generalization\\n')\n",
            "    display(Image(fig2_path))\n",
            "\n",
            "# Display final report\n",
            "report_path = 'results/WEEK1_FINAL_REPORT.md'\n",
            "if Path(report_path).exists():\n",
            "    print('\\n📄 Final Report (excerpt):\\n')\n",
            "    with open(report_path) as f:\n",
            "        lines = f.readlines()[:50]  # First 50 lines\n",
            "        display(Markdown(''.join(lines)))\n",
            "\n",
            "print('\\n✅ Training Complete! All results saved to Drive.')"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "Medical Robotics Week 1 Training",
                "provenance": [],
                "gpuType": "T4",
                "collapsed_sections": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook

def main():
    """Generate and save notebook"""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'notebooks'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'week1_training_colab.ipynb'
    
    print('Generating Colab training notebook...')
    notebook = generate_notebook()
    
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f'✅ Notebook generated: {output_file}')
    print(f'📊 Size: {output_file.stat().st_size / 1024:.1f} KB')
    print(f'📝 Cells: {len(notebook["cells"])}')
    print()
    print('📋 Next steps:')
    print('1. Upload to Google Drive')
    print('2. Open with Google Colaboratory')
    print('3. Runtime → Change runtime type → GPU')
    print('4. Runtime → Run all')

if __name__ == '__main__':
    main()
