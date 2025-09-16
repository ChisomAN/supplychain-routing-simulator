# Test Plan — RL Supply-Chain Routing Simulator

## Objective
Ensure the data product meets functional, usability, and reliability requirements before deployment.

---

## Test Scenarios

### 1. Data Loading
- **Input**: Upload valid CSV file.  
- **Expected**: File loads successfully, preview displayed.  
- **Input**: Upload invalid CSV.  
- **Expected**: Error displayed, app continues running.  
- **Input**: Load from valid URL.  
- **Expected**: Data loads and preview is shown.  

### 2. Data Exploration
- **Input**: Select numeric column for histogram.  
- **Expected**: Plot displays correctly.  
- **Input**: Select two columns for scatter plot.  
- **Expected**: Scatter plot renders with chosen axes.  

### 3. Data Cleaning
- **Action**: Apply normalization, IQR-based outlier removal.  
- **Expected**: Cleaned dataset preview shown.  
- **Action**: Drop rows with missing values.  
- **Expected**: Row count decreases accordingly.  

### 4. Modeling
- **Action**: Run A\* baseline.  
- **Expected**: Path length metrics displayed.  
- **Action**: Train RL model (toy).  
- **Expected**: Model trains for small timesteps without crashing.  

### 5. Results
- **Action**: Compare baseline vs. RL.  
- **Expected**: KPI metrics shown in JSON format.  

### 6. Reports
- **Action**: Click “Generate Report.”  
- **Expected**: Report is created (PDF if available, else TXT).  
- **Action**: Download report and cleaned CSV.  
- **Expected**: Files download successfully.  

### 7. Pipeline
- **Action**: Run full pipeline.  
- **Expected**: Synthetic data generated → cleaned → baseline run → report created.  

### 8. Help Tab
- **Action**: Open Help tab.  
- **Expected**: `HELP.md` renders inside the app.  

---

## Test Environment
- **Local**: macOS with Python 3.11, Streamlit 1.36  
- **Cloud**: Streamlit Cloud free tier  

---

## Notes
- Multi-user access tested on laptop + mobile simultaneously.  
- Heavy RL training intentionally limited for performance reasons.
