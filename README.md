# Fine Tuning Small Language Models On Healthcare Data

## To Run Training Scripts

It is recommended to run all training code as a python script (.py) in a separate space (running inside the same Python kernel though).

Example Code:

```bash
nohup /workspace/.venv/bin/python3 /workspace/scripts/medquad_fine_tune.py > training_resume.log 2>&1 &
```

To ensure better memory management from PyTorch and prevent memory fragmentation:  

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Without this, Python will "buffer" your training logs. You might see a blank log file for 30 minutes, then suddenly 5,000 lines appear at once. This forces real-time writes
```bash
export PYTHONUNBUFFERED=1
```

To clear VRAM from NVIDIA GPU in case of zombie processes use the following:

```bash
sudo fuser -k -9 /dev/nvidia*
```

_What This Does:_ `fuser` scans the Linux file system for any hidden, zombified, or container-isolated threads that are still hooked into the `/dev/nvidia0` device file and sends a hard kill signal to all of them simultaneously.

To kill a running process (for example a running Python script)

```bash
# Identify running process
ps aux | grep python

# Kill with PID
kill -9 <PID>
```