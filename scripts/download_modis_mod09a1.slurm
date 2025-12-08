#!/bin/bash
#SBATCH --job-name=modis_download
#SBATCH --output=logs/modis_download_%j.out
#SBATCH --error=logs/modis_download_%j.err
#SBATCH --time=72:00:00          # 72 hours (3 days) - adjust based on network speed
#SBATCH --mem=8G                 # 8GB memory for downloads and file handling
#SBATCH --cpus-per-task=1        # Single-threaded download (I/O bound)
#SBATCH --partition=common       # Duke Compute Cluster: common partition for general computing
#SBATCH --nodes=1                # Single node
#SBATCH --ntasks=1               # Single task
#SBATCH --mail-type=END,FAIL     # Email when job ends or fails
#SBATCH --mail-user=djm99@duke.edu

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Change to project root directory (adjust path if needed)
# Assumes script is run from project root, or adjust path accordingly
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT" || exit 1

# Load Python module (Duke Compute Cluster)
# Check available versions with: module avail python
# Common versions: python/3.8, python/3.9, python/3.10, python/3.11
# module load python/3.10

# Alternative: If you have a virtual environment set up, activate it instead:
# source venv/bin/activate
# or for conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# If using virtual environment:
source venv/bin/activate

# Verify Python and required packages
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found in project root!"
    echo "Please create .env file with LAADS_TOKEN before running this job."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory if it doesn't exist
mkdir -p data/raw/modis

# Run the MODIS download script
echo ""
echo "Starting MODIS MOD09A1 download..."
echo "Script: scripts/download_modis_mod09a1.py"
echo ""

python3 scripts/download_modis_mod09a1.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

