# Architecture Diagram

```mermaid
flowchart TB
    %% Client Layer
    subgraph CLIENTS["Client Layer"]
        REST["REST API\n(FastAPI :8000)"]
        UI["Streamlit UI\n(:8501)"]
        TG["Telegram Bot\n(long-polling)"]
        CRON["Cron Job\n(every 30 min)"]
    end

    %% Orchestrator
    subgraph ORCH["Orchestrator"]
        OA["OrchestratorAgent\n(facade)"]
        subgraph GRAPH["MultiAgentGraph (LangGraph StateGraph)"]
            SUP["Supervisor Node\n(LLM router)"]
            SYNTH["Synthesiser Node\n(merge outputs)"]
        end
    end

    %% Specialist Agents
    subgraph AGENTS["Specialist Agents"]
        CA["ConversationalAgent\n(greetings / small talk)"]
        PA["PropertySearchAgent\n(RAG-enabled)"]
        FA["FinanceAgent\n(stocks only)"]
    end

    %% Core Infrastructure
    subgraph CORE["Core Infrastructure"]
        LMG["LlmModelGraph\n(retrieve → generate)"]
        EMB["Embedder\nall-MiniLM-L6-v2\n384-dim"]
        SM["SessionManager\n(SQLite)"]
        LLM["Claude via Anthropic API\n(claude-opus-5-5)"]
    end

    %% MCP Tools
    subgraph MCP["MCP Tool Layer (mcp.json)"]
        YF["@fre4x/yahoo-finance\n(stocks server)\nyfin_get_quotes\nyfin_get_stock_info\nyfin_get_news\nyfin_get_recommendations\n+6 more tools"]
        FIN["@easysolutions906/mcp-finance\n(finance server)"]
    end

    %% Vector Store
    subgraph VS["Vector Store (persistence/)"]
        PC["PineconeStore\n(cloud / prod)"]
    end

    %% Data Ingestion
    subgraph INGEST["Data Ingestion"]
        CSV["CSV Pipeline\n(Polish housing data)"]
        SCRAPE["Web Scraper\n(BeautifulSoup + Selenium)"]
    end

    %% Observability
    LS["LangSmith\n(traces)"]

    %% Connections — Clients → Orchestrator
    REST --> OA
    UI --> REST
    TG --> OA
    CRON --> OA

    %% Orchestrator flow
    OA --> SUP
    SUP -->|route| CA
    SUP -->|route| PA
    SUP -->|route| FA
    CA --> SUP
    PA --> SUP
    FA --> SUP
    SUP -->|FINISH| SYNTH

    %% Agents → Core
    CA --> LMG
    PA --> LMG
    FA --> LMG

    %% Core internals
    LMG --> LLM
    LMG --> SM
    LMG --> EMB

    %% Core → Vector Store
    EMB --> PC

    %% Core → MCP Tools
    LMG -->|tool_calls| YF
    LMG -->|tool_calls| FIN

    %% Ingest → Vector Store
    CSV --> EMB
    SCRAPE --> EMB

    %% Observability
    LMG -.->|@traceable| LS

    %% Registry files
    AGENTSJSON["agents.json\n(agent registry)"] -.->|read at startup| ORCH
    MCPJSON["mcp.json\n(MCP registry)"] -.->|read at startup| MCP

    %% Styling
    classDef client fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef orchestrator fill:#7B4EA6,stroke:#4A2D6B,color:#fff
    classDef agent fill:#2E8B57,stroke:#1A5C38,color:#fff
    classDef core fill:#D4720B,stroke:#8B4A07,color:#fff
    classDef mcp fill:#C0392B,stroke:#7B2218,color:#fff
    classDef storage fill:#5D6D7E,stroke:#2C3E50,color:#fff
    classDef ingest fill:#1A7F7F,stroke:#0D4F4F,color:#fff
    classDef config fill:#888,stroke:#555,color:#fff,stroke-dasharray:5 5

    class REST,UI,TG,CRON client
    class OA,SUP,SYNTH orchestrator
    class CA,PA,FA agent
    class LMG,EMB,SM,LLM core
    class YF,FIN mcp
    class PC storage
    class CSV,SCRAPE ingest
    class AGENTSJSON,MCPJSON,LS config
```