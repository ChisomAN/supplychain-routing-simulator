
Contents:
- demo.ipynb … end-to-end load → EDA → cleaning → save artifacts → log run
- eda_cleaning.ipynb … compact walkthrough focused on exploration/cleaning
- model_stubs.ipynb … placeholders for A* and DQN next iteration
- sample_edges.csv … clean sample
- sample_edges_messy.csv … messy sample to demonstrate cleaning
- requirements.txt … Python dependencies
- artifacts/ … folder where cleaned outputs and logs are saved

How to run (locally):
1) (Optional) Create and activate a virtual environment.
2) Install dependencies:
   pip install -r requirements.txt
3) Open Jupyter and run demo.ipynb top to bottom.
4) Outputs are written to artifacts/datasets and logs to artifacts/logs.

Notes:
- This first iteration keeps modeling stubs only; baseline A* and DQN will be wired next iteration.
- If internet is restricted during grading, use sample_edges.csv for the file path demo.
