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
