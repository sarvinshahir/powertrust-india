
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

st.set_page_config(
    page_title="PowerTrust India Solar Intelligence",
    page_icon="☀️",
    layout="wide"
)

CHROMA_DIR = "/content/drive/MyDrive/PowerTrust-India/data/chroma_db"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def load_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name="india_solar", embedding_function=ef)

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY
    )

RAG_PROMPT = """You are an expert analyst for PowerTrust, helping assess distributed solar development in India.

Answer the question using ONLY the context provided below.
If the context does not contain enough information, say exactly:
"I cannot answer this from the available data. To answer this, we would need: [specify missing data]"

Do NOT hallucinate or make up numbers. Only cite what is in the context.

Context:
{context}

Question: {question}

Answer:"""

def retrieve(collection, query, n_results=5):
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = ""
    sources = []
    for doc, meta in zip(docs, metadatas):
        context += f"[Source: {meta['filename']}]\n{doc}\n\n"
        sources.append(meta["filename"])
    return context, list(set(sources))

def ask(collection, llm, question, chat_history=None):
    if chat_history:
        history_str = "\n".join([f"Q: {h['question']}\nA: {h['answer']}"
                                   for h in chat_history[-3:]])
        enriched = f"Previous conversation:\n{history_str}\n\nCurrent question: {question}"
    else:
        enriched = question
    context, sources = retrieve(collection, enriched)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return {"answer": response.content, "sources": sources}

collection = load_collection()
llm = load_llm()

# Header
st.title("☀️ PowerTrust — India Solar Intelligence")
st.caption("Open-data intelligence for distributed solar development in India")
st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Chat", "⚡ Risk Index", "📋 Data Audit"])

# ── TAB 1: DASHBOARD ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("India Solar Market — Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("### 45 GW\n**Solar Added FY26**\n\n*Record year*")
    col2.markdown("### 292 GW\n**Solar Target 2030**\n\n*NEP 2022-32*")
    col3.markdown("### 16 Lakh\n**Rooftop Installs**\n\n*PM Surya Ghar*")
    col4.markdown("### ₹2.5/kWh\n**Latest Auction**\n\n*SECI 2025*")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("CAPEX by Generation Type")
        capex_data = pd.DataFrame({
            "Category": ["Rooftop 1-10kW", "Rooftop 10-100kW", 
                        "Ground Mount <1MW", "Ground Mount >1MW", 
                        "Wind Onshore", "CCGT Gas"],
            "CAPEX (₹ Lakh/MW)": [5500, 4800, 4200, 3800, 6500, 4500],
            "Source": ["MNRE 2021-22", "MNRE 2021-22", "CERC 2024",
                      "SECI Auction", "CERC 2024", "CERC 2024"]
        })
        fig1 = px.bar(capex_data, x="Category", y="CAPEX (₹ Lakh/MW)",
                     color="Category", hover_data=["Source"],
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig1.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.subheader("Annual Solar Capacity Addition")
        capacity_data = pd.DataFrame({
            "Year": ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"],
            "GW": [5.5, 10.0, 13.5, 18.5, 24.3, 45.0]
        })
        fig2 = px.line(capacity_data, x="Year", y="GW",
                      markers=True,
                      color_discrete_sequence=["#FF6B35"])
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("State-wise RPO Targets (2024-25)")
        rpo_data = pd.DataFrame({
            "State": ["Maharashtra", "Karnataka", "Tamil Nadu", "Rajasthan", "Gujarat"],
            "RPO (%)": [17, 20, 18, 22, 19]
        })
        fig3 = px.bar(rpo_data, x="State", y="RPO (%)", color="State",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        st.subheader("Data Coverage by Dimension")
        coverage = pd.DataFrame({
            "Dimension": ["Cost & Economics", "Grid Access", "Subsidies & Policy",
                         "Utility Standards", "Public Approvals", "Unknown Risks"],
            "Coverage Score": [85, 45, 80, 70, 40, 75]
        })
        fig4 = px.bar(coverage, x="Coverage Score", y="Dimension",
                     orientation="h",
                     color="Coverage Score",
                     color_continuous_scale="RdYlGn")
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Six Dimension Summary")
    dims = {
        "1. Cost & Economics": "CAPEX ₹38-55 Lakh/MW. SECI 2025 auction: ₹2.5-3.2/kWh. IRENA India LCOE declined 89% since 2010.",
        "2. Grid Access": "CEA NEP targets 292 GW solar by 2030. Green Energy Corridors Phase II underway. Congestion highest in Tamil Nadu, Rajasthan.",
        "3. Subsidies & Policy": "PM Surya Ghar: ₹78,000 CFA for 3kW rooftop. PM-KUSUM for agricultural solar. Net metering up to 10kW without technical study.",
        "4. Utility Standards": "National RPO: 29.91% by 2029-30. Solar-specific RPO increasing annually. MERC 2024 regulations updated.",
        "5. Public Approvals": "No solar-specific EIA guidance. Central clearance via MoEF. SEIAA delays identified as bottleneck in NGT case.",
        "6. Unknown Unknowns": "600kt solar panel waste by 2030. Import duty changes as Change-in-Law risk. ALMM compliance creating supply constraints."
    }
    for dim, summary in dims.items():
        with st.expander(dim):
            st.write(summary)

# ── TAB 2: CHAT ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("💬 Solar Intelligence Chat")
    st.markdown("*Answers grounded in 60 documents from CERC, MNRE, SECI, CEA, IRENA and more*")
    st.info("This chat only answers from your downloaded documents. When data is unavailable, it says so explicitly — no hallucination.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    quick = st.selectbox("Quick queries", [
        "-- select --",
        "What is the benchmark CAPEX for rooftop solar installations in India?",
        "What are the net metering regulations for rooftop solar in India?",
        "What subsidies are available under PM Surya Ghar scheme?",
        "What data is missing that would most affect our solar CAPEX estimate?",
        "How do grid interconnection wait times vary across Indian states?",
        "What are the RPO targets for Maharashtra and Karnataka?"
    ])
    if quick != "-- select --":
        st.session_state.quick_q = quick

    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            st.caption(f"📄 Sources: {', '.join(msg['sources'][:3])}")

    col_input, col_btn = st.columns([5,1])
    with col_input:
        query = st.text_input("Ask about India solar — CAPEX, grid, policy, approvals...", key="chat_input")
    with col_btn:
        st.write("")
        send = st.button("Send")
    if not send:
        query = None

    if hasattr(st.session_state, "quick_q"):
        query = st.session_state.quick_q
        del st.session_state.quick_q

    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Searching 7,925 document chunks..."):
                result = ask(collection, llm, query, st.session_state.chat_history)
            st.write(result["answer"])
            st.caption(f"📄 Sources: {', '.join(result['sources'][:3])}")
        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"]
        })

# ── TAB 3: RISK INDEX ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("⚡ Solar Feasibility & Risk Index — India")
    st.markdown("*Scoring based on data extracted from public documents*")

    col1, col2 = st.columns([1, 1])

    with col1:
        scores = {
            "Cost & Economics": 72,
            "Grid Access": 45,
            "Policy Support": 78,
            "Utility Standards": 65,
            "Approval Process": 40,
            "Market Stability": 60
        }
        overall = int(sum(scores.values()) / len(scores))
        st.metric("Overall Feasibility Score", f"{overall}/100", "Moderate-High")

        for dim, score in scores.items():
            color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            st.markdown(f"{color} **{dim}**: {score}/100")
            st.progress(score / 100)

    with col2:
        categories = list(scores.keys())
        values = list(scores.values()) + [list(scores.values())[0]]
        categories_closed = categories + [categories[0]]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(255, 107, 53, 0.3)",
            line=dict(color="#FF6B35", width=2)
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="India Solar Feasibility Radar",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.subheader("State-level Risk Comparison")
    state_data = pd.DataFrame({
        "State": ["Maharashtra", "Karnataka", "Tamil Nadu", "Rajasthan", "Gujarat"],
        "Grid Risk": [65, 55, 75, 40, 45],
        "Policy Score": [72, 80, 70, 85, 78],
        "Land Risk": [70, 60, 65, 30, 35]
    })
    fig_state = px.scatter(state_data,
                          x="Grid Risk", y="Policy Score",
                          size="Land Risk", color="State", text="State",
                          title="State Risk Matrix (bubble size = land acquisition risk)",
                          color_discrete_sequence=px.colors.qualitative.Set1)
    fig_state.update_traces(textposition="top center")
    fig_state.update_layout(height=400)
    st.plotly_chart(fig_state, use_container_width=True)

# ── TAB 4: DATA AUDIT ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Data Availability Audit")
    st.markdown("*Required deliverable — transparency on sources, gaps, and decision impact*")

    st.success(f"✅ **60 documents parsed** | **7,925 chunks embedded** | **11 sources accessed**")

    st.subheader("Data Successfully Sourced")
    sourced = pd.DataFrame({
        "Source": ["CERC", "SECI", "MNRE", "IRENA", "Lazard",
                  "CEA", "NITI Aayog", "MERC", "MoEF/Parivesh",
                  "PRS Legislative Research", "PIB"],
        "Documents": [22, 4, 8, 1, 1, 4, 3, 4, 2, 4, 4],
        "Key Data": ["Tariff orders, RPO, grid charges", "Auction results, PPA templates",
                    "Benchmark costs, PM Surya Ghar", "Global LCOE trends",
                    "LCOE benchmarks", "NEP, transmission plans",
                    "Energy scenarios, state index", "MYT, RPO, net metering",
                    "EIA notification", "Parliamentary committee reports",
                    "Policy announcements, capacity data"]
    })
    st.write(sourced.to_markdown(index=False))

    st.subheader("Critical Data Gaps")
    gaps = pd.DataFrame({
        "Missing Data": ["Grid interconnection wait times",
                        "Open access queue data",
                        "State rooftop installation progress",
                        "Solar-specific public hearings",
                        "CAPEX benchmark post 2022"],
        "Source Attempted": ["POSOCO/Grid-India", "openaccessregistry.com",
                            "solarrooftop.gov.in", "MPCB, NGT",
                            "MNRE, CERC"],
        "Reason for Gap": ["Website inaccessible", "Website inaccessible",
                          "403 Forbidden", "No solar cases found",
                          "Latest data is 2021-22"],
        "Impact": ["Cannot quantify grid queue — key weakness",
                  "Cannot measure open access barriers",
                  "Cannot track PM Surya Ghar by state",
                  "Dimension 5 relies on indirect sources",
                  "CAPEX estimates may be 15-20% outdated"]
    })
    st.write(gaps.to_markdown(index=False))

    st.subheader("Model Update Methodology")
    st.markdown("""
    1. **Add new PDFs** to Drive folder → re-run `pdf_parser.py` + `embedder.py`
    2. **New countries** → replace India sources with equivalent country sources
    3. **Scheduled refresh** → CEA monthly reports and CERC orders can be scraped monthly
    4. **Version control** → ChromaDB collections versioned by date
    5. **Gap remediation** → when POSOCO/Grid-India comes online, re-run grid dimension
    """)
