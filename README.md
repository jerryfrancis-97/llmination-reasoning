# llmination-reasoning
Repo for LLM reasoning work

Steering code used from https://github.com/FlyingPumba/steering-thinking-models.

# Instructions
NB : **Make sure to create a new branch/ copy this branch {jerome-inference-dev} for your own work/exps, don't use the current branch for updates**

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