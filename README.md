markdown# ☀️ PowerTrust — India Solar Intelligence

> Open-data intelligence system for distributed solar development in India  
> Built for IE 7374 Generative AI Hackathon | Northeastern University | Spring 2026  
> Partner: PowerTrust (Kartik Naik, Data Scientist)

---

## 🎯 What This Does

A conversational intelligence platform that helps solar developers and investors assess distributed solar opportunities in India across **6 critical dimensions**:

1. **Cost & Economics** — CAPEX benchmarks, auction tariffs, LCOE trends
2. **Grid Access & Queue Dynamics** — Transmission capacity, congestion, NEP targets
3. **Subsidies, Incentives & Policy** — PM Surya Ghar, PM-KUSUM, net metering
4. **Utility Standards & Obligations** — RPO targets, REC framework, MERC regulations
5. **Public Comment & Approval Signals** — EIA process, SEIAA bottlenecks, NGT cases
6. **Unknown Unknowns** — Solar e-waste, ALMM supply risk, Change-in-Law, land conflicts

---

## 🏗️ Architecture
61 Government PDFs + Image Data
↓
PDF Parser (pdfplumber) + Smart Chunking
↓
ChromaDB Vector Store (3,414 chunks, all-MiniLM-L6-v2)
↓
Metadata-Filtered RAG Retrieval
↓
Groq LLM (llama-3.3-70b-versatile)
↓
Streamlit Dashboard (4 tabs)

---

## 📊 Data Sources

| Source | Documents | Key Data |
|--------|-----------|----------|
| CERC | 22 | Tariff orders, RPO regulations, AT orders 2025-26 |
| SECI | 4 | ISTS-XX auction results, PPAs (Rs2.5-2.6/kWh) |
| MNRE | 8 | Benchmark costs, PM Surya Ghar, net metering |
| IRENA | 1 | LCOE trends 2023, 89% cost decline since 2010 |
| Lazard | 1 | LCOE benchmarks across generation technologies |
| CEA | 4 | NEP 2022-32, transmission planning, capacity reports |
| NITI Aayog | 3 | Viksit Bharat energy scenarios, state climate index |
| MERC | 4 | MYT 2024, RPO REC Amendment 2024, Rooftop RE |
| MoEF | 2 | EIA Notification 2006 (Jan 2026), NGT OA 267/2025 |
| PRS Legislative | 4 | Solar performance report Dec 2025, PM-KUSUM |
| PIB | 4 | 45 GW FY26 record, ALMM expansion, recycling policy |

**Total: 61 documents | 3,414 chunks | 11 sources**

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector DB | ChromaDB (persistent, local) |
| LLM | Groq `llama-3.3-70b-versatile` |
| UI | Streamlit (4 tabs) |
| Parsing | pdfplumber |
| Chunking | Smart adaptive chunking (1500-2500 chars) |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.11
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
conda create -n powertrust python=3.11
conda activate powertrust
pip install chromadb langchain-groq langchain-core streamlit plotly pandas sentence-transformers pdfplumber tqdm
```

### Run

```bash
# Add your Groq API key to app.py
streamlit run app.py
```

---

## 🔍 Key Findings

| Finding | Source |
|---------|--------|
| India added record **45 GW** solar in FY2025-26 | PIB April 2026 |
| Solar target: **292,566 MW** by 2029-30 | CEA NEP 2022-32 |
| PM Surya Ghar: only **16%** of 1 crore target achieved | MNRE June 2025 |
| SECI auction: **Rs 2.50-2.60/kWh** discovered tariff | SECI ISTS-XX 2025 |
| **600,000 tonnes** solar e-waste projected by 2030 | PIB March 2026 |
| SEIAA Delhi had no functioning authority | NGT OA 267/2025 |
| No solar-specific EIA guidance manual exists | MoEF confirmed |
| GST increase 5%→12% = Change-in-Law risk in PPAs | CERC 219/MP/2023 |

---

## ⚠️ Data Gaps

| Missing Data | Reason | Impact |
|--------------|--------|--------|
| Grid interconnection wait times | POSOCO inaccessible | Dimension 2 partially blind |
| Open access queue data | openaccessregistry.com down | Cannot measure approval barriers |
| State rooftop progress | solarrooftop.gov.in 403 | Cannot track PM Surya Ghar by state |
| CAPEX benchmark post-2022 | MNRE latest is 2021-22 | Estimates may be 15-20% understated |
| Solar EIA guidance manual | Does not exist | Cannot assess approval complexity |

---

## 👥 Team

- **Sarvin Shahir, Mona Mahdavi** — Northeastern University, IE 7374 Generative AI
- **PowerTrust** — Industry Partner (Kartik Naik, Data Scientist)

---

## 📄 License

Academic project — IE 7374 Generative AI Hackathon, Northeastern University, Spring 2026
