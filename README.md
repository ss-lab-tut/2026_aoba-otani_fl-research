# SS-Lab Research Template

Minimal. Reproducible. Paper-Ready.

This template is designed for:

- B4 students starting research
- M1 students writing papers
- Ablation studies
- SOTA comparisons
- Multi-seed statistical evaluation

Philosophy:

Simple > Smart  
Reproducible > Fast  
Clear structure > Clever tricks  

------------------------------------------------------------

PROJECT STRUCTURE

research-template/
- README.md
- requirements.txt
- .gitignore
- LICENSE
- configs/                experiment configurations (JSON)
- src/                    reusable functions
- experiments/            runnable scripts
- results/                auto-created (NOT committed)

Note:
If you download the repository as ZIP from GitHub,
the folder name may include "-main". This is normal.

------------------------------------------------------------

WHAT GOES WHERE?

configs/
Each JSON file defines ONE experiment condition.

Examples:
- demo.json
- ablation_gat_off.json
- sota_baselineA.json

src/
Reusable code only.
No direct execution here.

experiments/
- run.py              run one config
- sweep.py            run all configs
- summarize.py        collect per-run results
- sweep_seeds.py      run multiple seeds
- summarize_seeds.py  compute mean/std

results/
Automatically created.
Never commit this folder.

------------------------------------------------------------

QUICK START

1) Create virtual environment

macOS / Linux:
python -m venv .venv
source .venv/bin/activate

Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

2) Install dependencies

pip install -r requirements.txt

If installation fails:
python -m pip install --upgrade pip

3) Run one experiment

python experiments/run.py configs/demo.json

------------------------------------------------------------

RUNNING MULTIPLE EXPERIMENTS

Run all configs once:

python experiments/sweep.py

Summarize per-run results:

python experiments/summarize.py

Output:
results/summary.csv

------------------------------------------------------------

MULTI-SEED STATISTICS (FOR PAPERS)

Run all configs with multiple seeds:

python experiments/sweep_seeds.py

Then summarize:

python experiments/summarize.py
python experiments/summarize_seeds.py

Final output:
results/summary_mean_std.csv

Meaning:

summary.csv
→ Each run (each seed) is one row

summary_mean_std.csv
→ Aggregated by:
  (experiment_name, group, variant)
  with mean and std

------------------------------------------------------------

EXPERIMENT CONFIG STRUCTURE

Example config:

{
  "experiment_name": "ablation_gat_off",
  "group": "ablation",
  "variant": "gat_off",
  "seed": 42
}

Fields:

experiment_name
  Unique name for the experiment

group
  proposed / ablation / sota / baseline

variant
  Short identifier of the condition

seed
  Random seed (fixed for reproducibility)

------------------------------------------------------------

WHAT IS ABLATION?

Ablation isolates components of your method.

Examples:
- Remove attention
- Replace GAT with MLP
- Change penalty weight
- Disable constraint term

Purpose:
Explain WHY your method works.

------------------------------------------------------------

WHAT IS SOTA COMPARISON?

SOTA comparison evaluates:

Your method vs existing methods.

Examples:
- Classical heuristic
- Published RL approach
- Public baseline implementation

Purpose:
Show performance advantage.

------------------------------------------------------------

REPRODUCIBILITY RULES (MANDATORY)

Every experiment must:

1) Use a config file
2) Fix random seed
3) Save config_used.json
4) Save metrics.json
5) Never manually edit results

If someone cannot reproduce your result,
the experiment is incomplete.

------------------------------------------------------------

RESULT STRUCTURE

Each run creates:

results/YYYYMMDD_HHMMSS_<experiment_name>/

Inside:
- config_used.json
- metrics.json

Example metrics.json:

{
  "experiment_name": "demo",
  "group": "proposed",
  "variant": "default",
  "seed": 42,
  "metrics": {
    "accuracy": 0.95,
    "reward": 123.4
  }
}

------------------------------------------------------------

ADDING A NEW EXPERIMENT

1) Create new JSON in configs/

Example:
configs/ablation_penalty_low.json

2) Run:

python experiments/run.py configs/ablation_penalty_low.json

Or run all:

python experiments/sweep.py

------------------------------------------------------------

ADDING NEW METRICS

In experiments/run.py,
modify run_experiment().

Return metrics as dictionary:

{
  "accuracy": value,
  "reward": value,
  "loss": value
}

They will automatically appear in summary.csv
and summary_mean_std.csv.

------------------------------------------------------------

COMMON MISTAKES

- Not activating virtual environment
- Editing results manually
- Forgetting to fix seed
- Mixing execution logic inside src/
- Saving outputs outside results/
- Running from wrong directory without activating environment

------------------------------------------------------------

GITHUB RULES (FOR STUDENTS)

Never commit directly to main.

Workflow:

1) Create branch:
   feature/<short-name>

2) Make changes

3) Open Pull Request

4) Merge

Examples:
- feature/add-train
- fix/readme-typo

Before PR:
Confirm that at least demo config runs successfully.

------------------------------------------------------------

DESIGN PHILOSOPHY

This template is:

- Not production engineering
- Not CI-heavy
- Not Docker-based

It is built for:

- Academic reproducibility
- Clear experiment comparison
- Student safety
- Sustainable research workflow

------------------------------------------------------------

FINAL ADVICE

If experiments become messy,
the problem is usually:

- Missing config separation
- Inconsistent metric naming
- No seed control
- Over-complicated structure

Keep experiments simple.
Keep metrics consistent.
Keep structure stable.

That is how research scales.
