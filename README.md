# RL Supply-Chain Routing Simulator

An interactive web-based data product for simulating and analyzing supply chain routing efficiency.  
The app combines **classical algorithms (A\*)** and **reinforcement learning (DQN)** to model, analyze, and optimize logistics networks.

---

## Live App
Deployed on **Streamlit Cloud**:  
[Supply Chain Routing Simulator](https://supplychain-routing-simulator-blmmuwwpt3m6dtucvgfc9d.streamlit.app/)

---

## Features
- **Load Data**: Import CSV files or generate synthetic graphs for routing analysis.  
- **Explore Data**: Visualize distributions, scatter plots, and summary statistics.  
- **Clean Data**: Normalize columns, remove outliers, and handle missing values.  
- **Modeling**:  
  - A\* baseline routing (distance, time, fuel).  
  - Optional RL (DQN) with simplified demo environment.  
- **Results**: KPI comparison between baseline and RL models.  
- **Reports**: Generate downloadable reports (PDF if ReportLab available, otherwise TXT).  
- **Pipeline**: One-click execution of the full workflow (synthetic → clean → model → report).  
- **Help**: Embedded user guide via `HELP.md`.  

---

## Run Locally
Clone the repo and install dependencies:

```bash
git clone https://github.com/ChisomAN/supplychain-routing-simulator.git
cd supplychain-routing-simulator

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
