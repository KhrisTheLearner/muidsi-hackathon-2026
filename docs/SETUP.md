# AgriFlow Setup Guide

Quick setup guide for MUIDSI Hackathon 2026.

## Prerequisites

- Python 3.10+
- Git
- Archia account (optional, for cloud deployment)

## Installation

```bash
# Clone repo
git clone <repo-url>
cd muidsi-hackathon-2026

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

```bash
# Required
DEFAULT_MODEL=priv-claude-sonnet-4-5-20250929

# Optional - for USDA NASS data
NASS_API_KEY=your_key_here

# Optional - for Archia cloud deployment
ARCHIA_TOKEN=your_token_here
ARCHIA_BASE_URL=https://registry.archia.app
```

## Database Setup

The SQLite database (`data/agriflow.db`) needs to be populated with USDA data.

### Required Tables

**1. food_environment** - USDA Food Environment Atlas
- Source: https://www.ers.usda.gov/data-products/food-environment-atlas/
- Columns: State, County, FIPS, FOODINSEC_15_17, POVRATE15, PCT_LACCESS_POP15, etc.
- ~3,100 rows (US counties)

**2. food_access** - USDA Food Access Research Atlas
- Source: https://www.ers.usda.gov/data-products/food-access-research-atlas/
- Columns: State, County, CensusTract, Urban, LowIncomeTracts, LATracts1, etc.
- ~70,000 rows (census tracts)

### Loading Data

```python
import sqlite3
import pandas as pd

# Load food environment data
conn = sqlite3.connect('data/agriflow.db')
food_env = pd.read_csv('path/to/food_environment.csv')
food_env.to_sql('food_environment', conn, if_exists='replace', index=False)

# Load food access data
food_access = pd.read_csv('path/to/food_access.csv')
food_access.to_sql('food_access', conn, if_exists='replace', index=False)

conn.close()
```

## Verification

```bash
# Test graph compilation
python -c "from src.agent.graph import create_agent; create_agent()"

# Test API server
uvicorn src.api.main:app --reload --port 8000
# Visit http://localhost:8000/api/health

# Test Archia skills (if configured)
python archia/setup_skills.py --verify
```

## Quick Start

```bash
# Option 1: Run local agent
python run_agent.py

# Option 2: Run API server
uvicorn src.api.main:app --reload --port 8000

# Option 3: Push to Archia cloud
python archia/setup_skills.py
```

## Troubleshooting

**Empty database results?**
- Load USDA data into `data/agriflow.db`

**NASS API errors?**
- Add NASS_API_KEY to .env
- Get key from: https://quickstats.nass.usda.gov/api

**Archia skills missing?**
- Check workspace in https://console.archia.app
- Regenerate API key in correct workspace
- Re-run `python archia/setup_skills.py`

## Next Steps

- See [RUNNING_AGENTS.md](RUNNING_AGENTS.md) for usage examples
- See [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) for system design
- See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment details
