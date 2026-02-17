# AgriFlow Database Schema

**File:** `data/agriflow.db` (SQLite)
**Version:** 2.0.3 | **Updated:** 2026-02-17

## Overview

AgriFlow uses a single SQLite database with two core tables sourced from USDA datasets, plus a dynamic schema that grows as the data ingestion agent adds new datasets.

```
data/agriflow.db
├── food_environment   (3,156 rows, 307 cols) — County-level food environment
├── food_access        (72,531 rows, 121 cols) — Census-tract food desert data
└── [ingest_*]         (dynamic) — Datasets added by the ingest agent
```

---

## Table: food_environment

**Source:** USDA Economic Research Service — Food Environment Atlas
**Granularity:** County (FIPS code)
**Coverage:** All 50 states + DC, 3,156 counties
**Missouri rows:** 115 counties

### Key Notes

- `FOODINSEC_*` and `VLFOODSEC_*` columns are **state-level aggregates** — all counties in a state share the same value. For Missouri all counties show 12.7% (2021-23 state average).
- `PCT_SNAP*` columns are also state-level.
- Best **county-level** risk indicators: `POVRATE21`, `PCT_LACCESS_POP*`, `MEDHHINC21`, store counts.

### Schema

```sql
CREATE TABLE food_environment (
    -- Identifiers
    FIPS            INTEGER,    -- 5-digit FIPS county code (e.g. 29510 = St. Louis City)
    State           TEXT,       -- 2-letter state abbreviation (e.g. 'MO')
    County          TEXT,       -- County name without "County" suffix

    -- Food Insecurity (STATE-LEVEL, uniform within state)
    FOODINSEC_18_20 REAL,       -- % food insecure households, 2018-20 avg (MO: 11.5)
    FOODINSEC_21_23 REAL,       -- % food insecure households, 2021-23 avg (MO: 12.7)
    VLFOODSEC_18_20 REAL,       -- % very low food security, 2018-20 (MO: 5.1)
    VLFOODSEC_21_23 REAL,       -- % very low food security, 2021-23 (MO: 5.8)
    CH_FOODINSEC_20_23 REAL,    -- Change in food insecurity 2020-23 (MO: 1.2)

    -- Poverty & Income (COUNTY-LEVEL)
    POVRATE21       REAL,       -- Poverty rate % (MO range: 5.3 - 23.9)
    CHILDPOVRATE21  REAL,       -- Child poverty rate % (MO range: 6.0 - 35.5)
    DEEPCHILDPOVRATE21 REAL,    -- Deep child poverty rate % (MO range: 1.0 - 20.4)
    DEEPPOVRATE21   REAL,       -- Deep poverty rate % (MO range: 1.8 - 14.3)
    PERPOV17_21     REAL,       -- Persistent poverty flag (0/1)
    MEDHHINC21      REAL,       -- Median household income $ (MO range: $36,683 - $92,029)

    -- Low Food Access — Population Counts (COUNTY-LEVEL)
    LACCESS_POP15   REAL,       -- Pop with low food access 2015 (MO range: 205 - 281,589)
    LACCESS_POP19   REAL,       -- Pop with low food access 2019
    LACCESS_LOWI15  REAL,       -- Low-income pop with low access 2015
    LACCESS_CHILD15 REAL,       -- Children with low food access 2015
    LACCESS_SENIORS15 REAL,     -- Seniors (65+) with low food access 2015
    LACCESS_HHNV15  REAL,       -- Households no vehicle with low access 2015
    LACCESS_SNAP15  REAL,       -- SNAP households with low access 2015

    -- Low Food Access — Percentages (COUNTY-LEVEL)
    PCT_LACCESS_POP15  REAL,    -- % pop with low access 2015 (MO range: 3.0 - 99.8)
    PCT_LACCESS_POP19  REAL,    -- % pop with low access 2019
    PCT_LACCESS_LOWI15 REAL,    -- % low-income pop with low access (MO range: 1.2 - 44.4)
    PCT_LACCESS_CHILD15 REAL,   -- % children with low access
    PCT_LACCESS_SENIORS15 REAL, -- % seniors with low access
    PCT_LACCESS_HHNV15 REAL,    -- % no-vehicle households with low access
    PCT_LACCESS_SNAP15 REAL,    -- % SNAP households with low access

    -- SNAP & Assistance (COUNTY-LEVEL counts; PCT_ versions are STATE-LEVEL)
    SNAPS17         REAL,       -- SNAP-authorized stores 2017 (MO range: 3 - 634)
    SNAPS23         REAL,       -- SNAP-authorized stores 2023 (MO range: 3 - 638)
    SNAPSPTH17      REAL,       -- SNAP stores per 1,000 pop 2017
    SNAPSPTH23      REAL,       -- SNAP stores per 1,000 pop 2023
    WICS16          REAL,       -- WIC-authorized stores 2016
    WICS22          REAL,       -- WIC-authorized stores 2022

    -- Grocery & Food Retail (COUNTY-LEVEL)
    GROC16          REAL,       -- Grocery stores 2016 (MO range: 1 - 181)
    GROC20          REAL,       -- Grocery stores 2020 (MO range: 3 - 185)
    GROCPTH16       REAL,       -- Grocery stores per 1,000 pop 2016
    GROCPTH20       REAL,       -- Grocery stores per 1,000 pop 2020
    SUPERC16        REAL,       -- Supercenters 2016 (MO range: 1 - 19)
    SUPERC20        REAL,       -- Supercenters 2020
    SUPERCPTH16     REAL,       -- Supercenters per 1,000 pop 2016
    CONVS16         REAL,       -- Convenience stores 2016 (MO range: 2 - 316)
    CONVS20         REAL,       -- Convenience stores 2020
    SPECS16         REAL,       -- Specialty food stores 2016
    FFR16           REAL,       -- Fast food restaurants 2016 (MO range: 1 - 837)
    FFR20           REAL,       -- Fast food restaurants 2020
    FSR16           REAL,       -- Full-service restaurants 2016 (MO range: 1 - 746)
    FSR20           REAL,       -- Full-service restaurants 2020

    -- School Nutrition Programs (COUNTY-LEVEL)
    PCT_NSLP17      REAL,       -- % students in lunch program 2017
    PCT_NSLP21      REAL,       -- % students in lunch program 2021
    PCT_FREE_LUNCH15 REAL,      -- % students on free lunch 2015
    PCT_REDUCED_LUNCH15 REAL,   -- % students on reduced lunch 2015
    PCT_SBP17       REAL,       -- % students in breakfast program 2017

    -- Health Outcomes (COUNTY-LEVEL)
    PCT_DIABETES_ADULTS15 REAL, -- % adults with diabetes 2015 (MO range: 8.0 - 13.9)
    PCT_DIABETES_ADULTS19 REAL, -- % adults with diabetes 2019 (MO range: 6.6 - 13.2)
    PCT_OBESE_ADULTS17 REAL,    -- % adults obese 2017

    -- Farmers Markets & Local Food (COUNTY-LEVEL)
    FMRKT13         REAL,       -- Farmers markets 2013
    FMRKT18         REAL,       -- Farmers markets 2018
    FMRKTPTH13      REAL,       -- Farmers markets per 1,000 pop 2013
    DIRSALES17      REAL,       -- Direct farm sales ($1,000) 2017
    CSA23           REAL,       -- Community-supported agriculture operations 2023

    -- Demographics (COUNTY-LEVEL)
    PCT_HISP20      REAL,       -- % Hispanic population 2020
    PCT_NHBLACK20   REAL,       -- % Non-Hispanic Black 2020
    PCT_NHWHITE20   REAL,       -- % Non-Hispanic White 2020
    PCT_18YOUNGER20 REAL,       -- % population under 18, 2020
    PCT_65OLDER20   REAL,       -- % population 65+, 2020

    -- Geography
    METRO23         REAL,       -- Metro area flag (1=metro, 0=rural) 2023
    POPLOSS15       REAL        -- Population loss flag 2015
    -- ... + 245 additional change/trend columns (PCH_*, PCT_* variants)
);
```

### Indexes
```sql
CREATE INDEX idx_fe_state  ON food_environment(State);
CREATE INDEX idx_fe_fips   ON food_environment(FIPS);
CREATE INDEX idx_fe_county ON food_environment(County);
```

### Example Queries

```sql
-- Top 10 most food-insecure MO counties by poverty rate
SELECT County, POVRATE21, PCT_LACCESS_POP15, MEDHHINC21, GROC20
FROM food_environment
WHERE State = 'MO'
ORDER BY POVRATE21 DESC
LIMIT 10;

-- Counties with high poverty AND low grocery access
SELECT County, POVRATE21, GROC20, GROCPTH20, SNAPS23
FROM food_environment
WHERE State = 'MO' AND POVRATE21 > 18 AND GROCPTH20 < 0.1
ORDER BY POVRATE21 DESC;
```

---

## Table: food_access

**Source:** USDA Economic Research Service — Food Access Research Atlas
**Granularity:** Census tract (CensusTract ID)
**Coverage:** All 50 states + DC, 72,531 tracts
**Missouri rows:** 1,391 census tracts

### Key Notes

- Provides **census-tract level** food desert classifications (more precise than food_environment).
- `LILATracts_*` flags identify food deserts by income + distance threshold.
- State column uses 2-letter codes (normalized from full names during load).
- County column has "County" suffix stripped for consistency with food_environment.

### Schema

```sql
CREATE TABLE food_access (
    -- Identifiers
    CensusTract     INTEGER,    -- 11-digit census tract FIPS code
    State           TEXT,       -- 2-letter state code (e.g. 'MO')
    County          TEXT,       -- County name (no "County" suffix)

    -- Geography & Population
    Urban           INTEGER,    -- Urban tract flag (1=urban, 0=rural)
    Pop2010         INTEGER,    -- Total population 2010
    OHU2010         INTEGER,    -- Occupied housing units 2010
    GroupQuartersFlag INTEGER,  -- High group quarters flag (1=yes)
    NUMGQTRS        REAL,       -- Group quarters population count
    PCTGQTRS        REAL,       -- % of pop in group quarters

    -- Food Desert Classification Flags (1=food desert, 0=not)
    LILATracts_1And10  INTEGER, -- LILA: 1-mile urban / 10-mile rural threshold
    LILATracts_halfAnd10 INTEGER,-- LILA: 0.5-mile urban / 10-mile rural
    LILATracts_1And20  INTEGER, -- LILA: 1-mile urban / 20-mile rural
    LILATracts_Vehicle INTEGER, -- LILA: vehicle access threshold
    HUNVFlag        INTEGER,    -- High share households no vehicle (1=yes)
    LowIncomeTracts INTEGER,    -- Low-income census tract flag (1=yes)

    -- Socioeconomic Indicators
    PovertyRate     REAL,       -- Tract poverty rate %
    MedianFamilyIncome REAL,    -- Median family income $

    -- Low-Access Flags (1 = has low-access population)
    LA1and10        INTEGER,    -- Low access at 1mi urban / 10mi rural
    LAhalfand10     INTEGER,    -- Low access at 0.5mi urban / 10mi rural
    LA1and20        INTEGER,    -- Low access at 1mi urban / 20mi rural
    LATracts_half   INTEGER,    -- Low-access tract at 0.5mi threshold
    LATracts1       INTEGER,    -- Low-access tract at 1mi threshold
    LATracts10      INTEGER,    -- Low-access tract at 10mi threshold
    LATracts20      INTEGER,    -- Low-access tract at 20mi threshold
    LATractsVehicle_20 INTEGER, -- Low-access by vehicle at 20mi

    -- Low-Access Population Counts (by distance threshold)
    LAPOP1_10       REAL,       -- Pop with low access: 1mi/10mi
    LAPOP05_10      REAL,       -- Pop with low access: 0.5mi/10mi
    LAPOP1_20       REAL,       -- Pop with low access: 1mi/20mi
    LALOWI1_10      REAL,       -- Low-income pop with low access: 1mi/10mi
    LALOWI05_10     REAL,       -- Low-income pop with low access: 0.5mi/10mi
    LALOWI1_20      REAL,       -- Low-income pop with low access: 1mi/20mi

    -- Low-Access at 0.5mi Threshold by Demographic
    lapophalf       REAL,       -- Total pop with low access at 0.5mi
    lapophalfshare  REAL,       -- % of tract pop with low access at 0.5mi
    lalowihalf      REAL,       -- Low-income pop with low access at 0.5mi
    lakidshalf      REAL,       -- Children with low access at 0.5mi
    laseniorshalf   REAL,       -- Seniors with low access at 0.5mi
    lawhitehalf     REAL,       -- White pop with low access at 0.5mi
    lablackhalf     REAL,       -- Black pop with low access at 0.5mi
    laasianhalf     REAL,       -- Asian pop with low access at 0.5mi
    lahisphalf      REAL,       -- Hispanic pop with low access at 0.5mi
    lahunvhalf      REAL,       -- No-vehicle households with low access at 0.5mi
    lasnaphalf      REAL,       -- SNAP households with low access at 0.5mi

    -- Low-Access at 1mi Threshold by Demographic
    lapop1          REAL,       -- Total pop with low access at 1mi
    lapop1share     REAL,       -- % of tract pop with low access at 1mi
    lalowi1         REAL,       -- Low-income pop with low access at 1mi
    lakids1         REAL,       -- Children with low access at 1mi
    laseniors1      REAL,       -- Seniors with low access at 1mi
    lawhite1        REAL,       -- White pop with low access at 1mi
    lablack1        REAL,       -- Black pop with low access at 1mi
    lahisp1         REAL,       -- Hispanic pop with low access at 1mi
    lahunv1         REAL,       -- No-vehicle households with low access at 1mi
    lasnap1         REAL,       -- SNAP households with low access at 1mi

    -- Low-Access at 10mi Threshold by Demographic
    lapop10         REAL,       -- Total pop with low access at 10mi
    lapop10share    REAL,       -- % of tract pop with low access at 10mi
    lalowi10        REAL,       -- Low-income pop with low access at 10mi
    lakids10        REAL,       -- Children with low access at 10mi
    laseniors10     REAL,       -- Seniors with low access at 10mi
    lahisp10        REAL,       -- Hispanic pop with low access at 10mi
    lahunv10        REAL,       -- No-vehicle households with low access at 10mi
    lasnap10        REAL,       -- SNAP households with low access at 10mi

    -- Tract Demographic Totals
    TractLOWI       REAL,       -- Total low-income population in tract
    TractKids       REAL,       -- Total child population in tract
    TractSeniors    REAL,       -- Total senior population in tract
    TractWhite      REAL,       -- Total White population in tract
    TractBlack      REAL,       -- Total Black population in tract
    TractAsian      REAL,       -- Total Asian population in tract
    TractHispanic   REAL,       -- Total Hispanic population in tract
    TractHUNV       REAL,       -- Total no-vehicle households in tract
    TractSNAP       REAL        -- Total SNAP households in tract
    -- ... + additional share columns (*share suffix)
);
```

### Indexes
```sql
CREATE INDEX idx_fa_state  ON food_access(State);
CREATE INDEX idx_fa_county ON food_access(County);
CREATE INDEX idx_fa_tract  ON food_access(CensusTract);
```

### Example Queries

```sql
-- All food desert census tracts in Missouri (LILA at 1mi/10mi standard)
SELECT CensusTract, County, Urban, Pop2010, PovertyRate, LILATracts_1And10
FROM food_access
WHERE State = 'MO' AND LILATracts_1And10 = 1
ORDER BY PovertyRate DESC;

-- Rural MO tracts with high low-income low-access population
SELECT County, CensusTract, LALOWI1_10, lalowi1share, PovertyRate
FROM food_access
WHERE State = 'MO' AND Urban = 0 AND LALOWI1_10 > 500
ORDER BY LALOWI1_10 DESC;

-- Join with food_environment for county-level context
SELECT fe.County, fe.POVRATE21, fe.MEDHHINC21,
       COUNT(fa.CensusTract) as total_tracts,
       SUM(fa.LILATracts_1And10) as food_desert_tracts,
       AVG(fa.PovertyRate) as avg_tract_poverty
FROM food_environment fe
JOIN food_access fa ON fe.State = fa.State AND fe.County = fa.County
WHERE fe.State = 'MO'
GROUP BY fe.County
ORDER BY food_desert_tracts DESC
LIMIT 10;
```

---

## Dynamic Tables (added by ingest agent)

The `agriflow-ingest` agent automatically adds new datasets as they are requested. Each new table is prefixed and documented here as it is created.

| Table | Source | Description | Added |
|-------|--------|-------------|-------|
| _(populated dynamically)_ | | | |

### Naming Convention

Ingest agent tables follow this pattern:
```
{source_tag}_{description}_{year}
```

Examples:
- `nass_corn_yields_2024` — USDA NASS crop yield data
- `census_poverty_2022` — Census poverty estimates
- `fema_disasters_2023` — FEMA disaster declarations
- `bls_unemployment_2024` — BLS county unemployment rates

---

## Reload Commands

```bash
# Reload core USDA datasets from raw CSVs
python -m src.data_pipeline.load_atlas

# Check current database state via agent
python run_agent.py "List all tables in the database and their row counts"

# Direct SQLite inspection
python -c "
import sqlite3
conn = sqlite3.connect('data/agriflow.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
for (t,) in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    cols = len(conn.execute(f'PRAGMA table_info({t})').fetchall())
    print(f'{t}: {n:,} rows, {cols} cols')
conn.close()
"
```

---

## Data Sources Reference

| Dataset | Source | Update Frequency | Level |
|---------|--------|-----------------|-------|
| Food Environment Atlas | USDA ERS | Annual | County |
| Food Access Research Atlas | USDA ERS | Every 5 years | Census Tract |
| NASS Quick Stats | USDA NASS | Weekly (growing season) | County/State |
| Census ACS 5-Year | US Census | Annual | County/Tract |
| FEMA Disaster Declarations | FEMA | Real-time | County |
| BLS LAUS | Bureau of Labor Statistics | Monthly | County |

---

**Schema Version:** 2.0.3
**Generated:** 2026-02-17
**Tool:** `python -m src.data_pipeline.load_atlas`
