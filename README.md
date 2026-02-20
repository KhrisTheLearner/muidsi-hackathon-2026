# 🌾 AgriFlow — Food Supply Chain Intelligence Agent

**MUIDSI Hackathon 2026 | Agriculture/Plant Track**
**Team:** Pierce (AI/ML Lead), Alfiya (Data Pipeline + EDA), Suyog (Feature Engineering + ML), Christophe (Problem Framing + Pitch)

---

## What is AgriFlow?

AgriFlow is an agentic AI system that helps food distribution planners optimize where to send resources by reasoning across crop supply data, weather disruptions, and community food access needs.

**Tagline:** *"Ask your supply chain anything."*

---

## Project Structure

```
agriflow/
├── data/
│   ├── raw/                  # Original datasets (DO NOT MODIFY)
│   ├── processed/            # Cleaned, transformed data
│   └── external/             # Any supplementary data
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── src/
│   ├── agent/                # LangChain agent logic, Archia integration
│   ├── data_pipeline/        # Data loading, cleaning, merging scripts
│   ├── models/               # ML model training, evaluation, prediction
│   ├── features/             # Feature engineering scripts
│   └── visualization/        # Charts, maps, and visual outputs
├── frontend/                 # AgriFlow React interface
├── docs/                     # Dataset documentation, pitch scripts
├── tests/                    # Unit tests
├── .env.example              # Template for environment variables
├── .gitignore
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Datasets

| Dataset | Source | Level | Format | Status |
|---------|--------|-------|--------|--------|
| Food Environment Atlas | USDA ERS | County | Excel/CSV | 📥 To download |
| Food Access Research Atlas | USDA ERS | Census tract | Excel | 📥 To download |
| NASS Quick Stats | USDA NASS | State/County | API (JSON) | 🔑 Need API key |

**Download links:**
- Food Environment Atlas: https://www.ers.usda.gov/data-products/food-environment-atlas/data-access-and-documentation-downloads
- Food Access Research Atlas: https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data
- NASS API: https://quickstats.nass.usda.gov/api (register for free API key)

**⚠️ Place raw data files in `data/raw/` — never modify originals.**

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/muidsi-hackathon-2026-ipg.git
cd muidsi-hackathon-2026-ipg
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Archia setup
```bash
export ARCHIA_TOKEN="your_archia_token_here"
# Get token from console.archia.app → MUIDSI Hackathon 2026 workspace → API Keys
```

---

## Role Assignments (Mapped to Rubric)

| Person | Role | Rubric Target | Key Deliverables |
|--------|------|---------------|------------------|
| **Pierce** | Agent architecture + core ML | Model Development (30%) | Agent logic, prediction model, Archia integration |
| **Suyog** | Feature engineering + evaluation | Feature Engineering (20%) | Feature selection, transformation, model metrics |
| **Alfiya** | Data pipeline + EDA | EDA (10%) | Data cleaning, exploration notebooks, visualizations |
| **Christophe** | Problem framing + pitch | Problem (5%) + Clarity (5%) | Problem statement, demo script, video production |

---

## Scoring Rubric

| Category | Weight |
|----------|--------|
| Problem Definition | 5% |
| Social Good Impact | 5% |
| EDA | 10% |
| **Feature Engineering** | **20%** |
| **Model Development** | **30%** |
| Evaluation Metrics | 10% |
| Clarity & Structure | 5% |
| Team Participation | 5% |
| Methodological Novelty | 10% |

**70% of the score is technical execution.**

---

## Timeline

| Milestone | Date |
|-----------|------|
| ✅ Kickoff | Fri Feb 13 |
| 🔨 Build | Sat Feb 14 – Mon Feb 16 |
| 🎬 **Round 1 Video Due** | **Tue Feb 17, 11:59 PM** |
| 📊 Results Announced | Wed Feb 18, 8:00 PM |
| 🏆 Finals (if selected) | Thu Feb 20, 1:00–5:00 PM |

---

## Git Workflow

- `main` — stable, working code only
- `dev` — integration branch
- Feature branches: `feature/your-name/description` (e.g., `feature/pierce/agent-setup`)
- Pull requests to `dev`, then merge to `main` when stable
- **Commit often, push daily**
