# llmination-reasoning
Repo for LLM reasoning work


## Overview

This repository contains a comprehensive framework for evaluating and analyzing reasoning capabilities in Large Language Models (LLMs). The framework includes tools for running reasoning evaluations, analyzing model behavior, and processing mathematical problem datasets.

## Setup

### Environment Variables
First, fill the `.env` file in the root directory with your API keys:

## Running the Framework

This repository provides several scripts and notebooks for evaluating and analyzing the reasoning capabilities of LLMs, especially on mathematical and logic problems. Below is a guide to the main files and how to use them.

Installing libraries and activating environment
PIP
1. Set up a virtual environment by running: `python3 -m venv simple_env`
2. Activate the environment and install dependencies with: `pip3 install -r requirements.txt`
3. TO see all libs, use `pip3 list`

CONDA
1. Create a conda environment from the provided YAML file (e.g., `environment.yml`):
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the new environment (replace `your_env_name` with the name specified in the YAML file, or check the `name:` field in `environment.yml`):
   ```bash
   conda activate your_env_name
   ```

3. To list all available conda environments:
   ```bash
   conda env list
   ```

To run the main scripts, use the following commands:
Use the follwing,

a. `python3 data/fixing_problems.py` : TO get perplexmath dataset
b. `python3 experiments/reasoning_evaluation.py` : TO get experiments results
c. Use model_behavior_analysis.ipynb to generate analysis for tables

---

### 1. `experiments/reasoning_evaluation.py`

**Purpose:**  
This is the main script for running large-scale reasoning evaluations on LLMs. It supports multiple models and APIs (e.g., Gemini, Groq, OpenRouter) and processes datasets of math/logic problems, extracting and analyzing the model's reasoning chains.

**Key Features:**
- Loads datasets (e.g., PerplexMath, custom JSON/CSV files).
- Queries LLMs with each problem, capturing both the final answer and the step-by-step reasoning.
- Annotates and classifies reasoning steps (e.g., recall, reasoning, exploration, uncertainty, computation).
- Computes detailed metrics and saves results in CSV/JSON formats.
- Supports batch runs and per-model result files.

**How to Run:**
1. **Prepare your environment:**
   - Install dependencies: `pip install -r requirements.txt`
   - Set your API keys in `.env` (see above).

3. **Customizing:**
   - To use a different dataset, change the path in the `framework.run()` call.
   - To use different models/APIs, edit the `models` list in the script.

4. **Results:**
   - Results are saved in the `results/` directory, organized by run and model.
   - Each run produces CSV and JSON files with detailed annotations and metrics.

---

### 2. `notebooks/model_behavior_analysis.ipynb`

**Purpose:**  
A Jupyter notebook for in-depth analysis and visualization of the results produced by `reasoning_evaluation.py`.

**Key Features:**
- Loads result files (CSV/JSON) from the `results/` directory.
- Provides summary statistics, plots, and breakdowns of reasoning types, confidence, and model performance.
- Allows for custom filtering and exploration of model outputs.

---

### 3. `experiments/finding_answers.py`

**Purpose:**  
A utility script for extracting and post-processing final answers from model outputs, especially when outputs are noisy or inconsistently formatted.

**Key Features:**
- Parses LLM responses to reliably extract the final answer.
- Can be used as a standalone script or imported as a module in other workflows.

**How to Run:**








# Extra Instructions

- For lambda compute setup
1. Start an instance and open the jupyter lab
2. Open the terminal in jupyter lab
3. Use the command export to add environment variables

export GIT_TOKEN=< from GIT Personal Access token, while creating select the repo scope>
export OPENROUTER_API_KEY=<from openrouter.com API>
export GEMINI_API_KEY=<from google ai studio>

or create a .env file.

4. Take the file deploy-llm-exp.sh from the repo and place it in the gpu's jupyter directory.
5. Run chmod u+x deploy-llm-exp.sh on terminal
6. Run ./deploy-llm-exp.sh on terminal



---

## ⭐️ If You Like This Repo...

If you find this repository helpful, please consider giving it a **star**! Your support helps others discover the project and motivates us to keep improving it.

## 🤝 Connect With Us

For questions, feedback, or collaboration opportunities, feel free to reach out to the authors. You can open an [issue](https://github.com/your-repo/issues) on GitHub or connect directly via email (see author profiles in the repository). We're always happy to discuss ideas, improvements, or research collaborations!

---
