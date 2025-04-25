# Hospital Transfer Network Analysis (SI 507 Final Project)

This project uses 2022–2023 hospital transfer data from California to model inter-hospital patient flows as a directed network. The CLI tool supports transfer path tracing, centrality analysis, and hospital similarity lookup.

## Features

- Network of ZIP codes and hospital facilities
- Static analysis by disease category and facility volume
- Cosine similarity clustering of similar hospitals
- CLI interface with:
  - Shortest path queries
  - Hospital stats
  - Centrality listing
  - Similarity search

## File Structure

- `hospital_transfer_analysis.py` – Main analysis and CLI script
- `data/` – All cleaned CSV files used in the project
- `requirements.txt` – Required packages
- `README.md` – Project description

## How to Run

Make sure you have Python 3 installed. Then install required packages:

```bash
pip install -r requirements.txt