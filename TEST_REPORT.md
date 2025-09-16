# Test Report — RL Supply-Chain Routing Simulator

## Summary
Testing was performed locally and on Streamlit Cloud to validate functionality, usability, and stability.  
All major features passed with minor notes on RL training performance (expected due to limited resources).  

---

## Test Results

### Data Loading
- Valid CSV uploaded successfully.  
- Invalid CSV triggered error without crashing.  
- URL loading worked with public CSV links.  
**Passed**

### Data Exploration
- Histogram plots displayed via Plotly.  
- Scatter plots generated correctly.  
**Passed**

### Data Cleaning
- Normalization and IQR filtering applied correctly.  
- Dropping missing rows reduced dataset size.  
**Passed**

### Modeling
- A\* baseline ran and returned expected path metrics.  
- RL (DQN) training worked for small timesteps (200) but slowed on higher settings.  
**Passed (with limits)**

### Results
- KPI evaluation compared baseline and RL outputs successfully.  
**Passed**

### Reports
- TXT reports generated successfully.  
- PDF generation worked when ReportLab was installed.  
- Download buttons functioned correctly.  
**Passed**

### Pipeline
- Full pipeline executed: synthetic → clean → baseline → report.  
**Passed**

### Help Tab
- `HELP.md` rendered properly inside the app.  
**Passed**

---

## Deployment Notes
- Verified live app link: [Streamlit Cloud Deployment](https://supplychain-routing-simulator-blmmuwwpt3m6dtucvgfc9d.streamlit.app/)  
- Tested with multiple simultaneous users (laptop + phone).  
- Each session was isolated, no data conflicts observed.  

---

## Conclusion
The app meets functional requirements, passes all major test scenarios, and is ready for academic submission.  
