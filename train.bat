@echo off

REM Change directory to the specified path
cd /d D:\Aj_Aof_Work\OCT_Disease\Grading_new\DiabeticRetinopathy_grading

REM Open a new Command Prompt window to run TensorBoard with Conda environment activated
start cmd /k "conda activate deep_torch && tensorboard --logdir aptos2019_logs --host localhost --port 6006"

REM Activate Conda environment
call conda activate deep_torch

REM Run the Python script
python train_pl_3.py

REM Keep the command prompt open if the script finishes or encounters an error
pause