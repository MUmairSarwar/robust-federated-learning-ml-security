# GitHub From Zero

1. On GitHub, create a **public** repository named `robust-federated-learning-ml-security`. Because this project already contains a README, `.gitignore`, and MIT license, leave GitHub's initialization boxes unchecked.
2. In a terminal, run:

```bash
git clone REPOSITORY_URL
cd robust-federated-learning-ml-security
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_centralized.py
python src/main.py --config experiments/baseline_fedavg.yaml
python src/main.py --config experiments/label_poisoning_fedavg.yaml
python src/main.py --config experiments/backdoor_fedavg.yaml
python src/main.py --config experiments/fuzzy_reliability_defense.yaml
git add .
git commit -m "Initial robust federated learning security project"
git branch -M main
git push origin main
```

On Windows PowerShell, activation is:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt, use `.venv\Scripts\activate.bat`.

Run the quick check first if needed:

```bash
python src/main.py --config experiments/quick_demo.yaml --smoke-test
```

After all full experiments finish, combine the generated CSV values in the comparison template, generate comparison figures, add selected plots to the README, commit the measured outputs, and replace `[GitHub link]` in your LinkedIn draft.

