# Security & Privacy Notes

- **Local-first**: No external databases or APIs are required for first iteration. All artifacts are local under `artifacts/`.
- **PII**: Do not upload personally identifiable information. Synthetic data is provided for demos.
- **Access**: Single-user desktop execution. Instructors run locally with their own environment.
- **Malicious files**: If a file/URL load fails or appears suspicious, the app halts processing and logs to `artifacts/logs/runs.jsonl`.
- **Secrets**: No API keys are used in v1. If added later, use environment variables and avoid committing secrets.
- **Reports**: Generated files include minimal metadata. Verify target folder before sharing outside your machine.
