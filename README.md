AI Dataset Review Agent (GitLab AI Hackathon)


🔍 Overview

AI Dataset Review Agent is an automated GitLab pipeline agent that analyzes datasets and generates actionable machine learning insights for developers.

It helps teams quickly understand dataset quality, structure, and ML readiness during the development workflow.


Problem

In ML projects, developers often:

manually inspect datasets
miss data quality issues
waste time understanding structure

There is no automated dataset validation in CI/CD workflows.


Solution

This project introduces an AI-powered GitLab Agent that:

detects dataset files automatically
analyzes structure and quality
identifies ML problem type
generates insights and recommendations

All triggered automatically inside a GitLab pipeline.


Trigger:
Push to repository (GitLab pipeline starts)
↓  
Action:
Agent scans repository → finds CSV dataset  
↓  
Processing:
- dataset analysis (pandas)
- target detection
- problem type detection
- correlation insights
↓  
Output:
analysis_report.txt (artifact)


Features

Automatic dataset analysis
ML problem detection (Classification / Regression)
Missing value detection
Correlation insights
Smart recommendations
Fully automated via GitLab CI/CD


Tech Stack
Python
Pandas
GitLab CI/CD
YAML pipelines


Project Structure

ml-dashboard/
│
├── agent.py
├── data/
│   └── sample.csv
├── .gitlab-ci.yml
├── analysis_report.txt (generated)
└── README.md


Example Output

=== AI Dataset Review Agent Report ===

Analyzed File: data/sample.csv
Execution Context: GitLab Pipeline Trigger

Overview:
- Rows: 5
- Columns: 4

Data Quality:
- No missing values detected

ML Insights:
- Problem Type: Classification
- Target Column: target

Key Observations:
- Strong correlation between age and salary

Recommendations:
- Handle missing values before training
- Suitable for classification models

Final Assessment:
- Dataset is suitable for Classification tasks


Impact

This agent reduces manual effort in:

dataset validation
ML workflow setup
early-stage debugging

It integrates directly into developer workflows, improving productivity and reliability.


Future Improvements
GitLab comment integration (auto-post report on MR)
support for multiple datasets
advanced AI insights using LLMs
visualization support


Getting Started
Clone repo
Add dataset (.csv)
Push to GitLab
Pipeline runs automatically
Download artifact (analysis_report.txt)


Hackathon Category Fit
GitLab Duo Agent Platform ✅
Automated SDLC workflow ✅
Trigger + Action Agent ✅