# Security Policy

## Data Security
- Data files uploaded to the app are handled in-memory or stored temporarily in the `artifacts/` directory.  
- No sensitive or personal user data is collected.  
- Sample datasets provided (`sample_edges.csv`) are anonymized and safe for demonstration purposes.  

## Access Control
- The Streamlit app runs on a public URL.  
- All sessions are isolated per user, ensuring one user's data does not affect another's session.  
- For private deployments, GitHub authentication or Streamlit team accounts can be enabled.  

## Vulnerability Handling
- If a security issue is found, please open a GitHub issue or contact the developer directly.  
- Known risks include:  
  - Uploading malformed CSV files (handled with error catching).  
  - Heavy RL training (restricted to toy settings to avoid abuse).  

## Contact
Report security concerns to:  
**Chisom “Chizzy” Atulomah** — nielatulomah28@gmail.com
