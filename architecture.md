# Architecture Diagram

```mermaid
flowchart TB
    %% Client Layer
    subgraph CLIENTS["Client Layer"]
        REST["REST API\n(FastAPI :8000)"]
        UI["Streamlit UI\n(:8501)"]
        TG["Telegram Bot\n(text + voice notes)"]
        CRON["Cron Job\n(every 30 min)"]
    end

    %% Standalone services
    subgraph SERVICES["Standalone Services"]
        WAL["WhatsAppListener\n(agents.whatsapp.main)"]
        MM["MailMonitor\n(agents.mails.main)"]
    end

    %% Orchestrator
    subgraph ORCH["Orchestrator"]
        OA["OrchestratorAgent\n(facade, on_route)"]
        MEM["ConversationMemory\n(last 3 turns)"]
        subgraph GRAPH["MultiAgentGraph (LangGraph, SQLite checkpointer)"]
            ROUTER["Router Node\n(clean + plan, 1 LLM call)"]
            SYNTH["Synthesiser Node\n(passthrough or merge)"]
            CONF["Confirmation Node\n(interrupt / resume)"]
        end
        GUARD["Output Guard\n(strips tool-call markup)"]
    end

    %% Specialist Agents
    subgraph AGENTS["Specialist Agents (agents.json)"]
        CA["ConversationalAgent"]
        PA["PropertySearchAgent\n(RAG)"]
        FA["FinanceAgent\n(stocks MCP)"]
        WA["WhatsAppAgent\n(RAG, short replies)"]
        MA["MailAgent"]
        EA["ExpenditureAgent\n(in-process tools)"]
        FBR["FbrTaxAgent\n(Claude Code session)"]
    end

    %% Core Infrastructure
    subgraph CORE["Core Infrastructure"]
        LMG["LlmModelGraph\n(retrieve → generate, tool loop)"]
        EMB["Embedder\nall-MiniLM-L6-v2"]
        SM["SessionManager\n(sessions.db)"]
        LLM["create_llm()\nChatAnthropic or ChatClaudeCode"]
        WHISPER["Transcriber\n(faster-whisper)"]
    end

    %% Agent-owned stores and tools
    LEDGER["expenses.db\n(SQLite ledger)"]
    CHROME["Claude in Chrome\n+ fbr-iris-filing skill"]

    %% MCP Tools
    subgraph MCP["MCP Tool Layer (mcp.json)"]
        YF["@fre4x/yahoo-finance\n(stocks server)"]
        FIN["@easysolutions906/mcp-finance\n(finance server)"]
    end

    %% Vector Store
    subgraph VS["Vector Store"]
        PC["PineconeStore"]
    end

    %% External delivery
    GREEN["Green API\n(WhatsApp)"]
    IMAP["IMAP inboxes"]
    LS["LangSmith\n(traces)"]

    %% Clients → Orchestrator
    REST --> OA
    UI --> REST
    TG --> OA
    TG --> WHISPER
    CRON --> OA

    %% Orchestrator flow
    OA --> ROUTER
    MEM --> ROUTER
    ROUTER -->|plan, in order| AGENTS
    AGENTS --> GUARD
    GUARD --> SYNTH
    SYNTH -->|pending item| CONF
    CONF -->|confirm / cancel / correction / unrelated| AGENTS
    SYNTH --> OA

    %% Agents → Core
    CA --> LMG
    PA --> LMG
    FA --> LMG
    WA --> LMG
    MA --> LMG
    EA --> LMG
    EA --> LEDGER
    FBR --> CHROME

    %% Core internals
    LMG --> LLM
    LMG --> SM
    LMG --> EMB
    EMB --> PC
    LMG -->|tool calls| YF
    LMG -->|tool calls| FIN

    %% Standalone services
    WAL --> GREEN
    WAL --> LLM
    MM --> IMAP
    MM --> LLM
    MM --> GREEN

    %% Observability and registries
    LMG -.->|@traceable| LS
    OA -.->|orchestrator.request| LS
    AGENTSJSON["agents.json\n(agent registry)"] -.->|read at startup| ORCH
    MCPJSON["mcp.json\n(MCP registry)"] -.->|read at startup| MCP

    %% Styling
    classDef client fill:#4A90D9,stroke:#2C5F8A,color:#fff
    classDef orchestrator fill:#7B4EA6,stroke:#4A2D6B,color:#fff
    classDef agent fill:#2E8B57,stroke:#1A5C38,color:#fff
    classDef core fill:#D4720B,stroke:#8B4A07,color:#fff
    classDef mcp fill:#C0392B,stroke:#7B2218,color:#fff
    classDef storage fill:#5D6D7E,stroke:#2C3E50,color:#fff
    classDef service fill:#1A7F7F,stroke:#0D4F4F,color:#fff
    classDef config fill:#888,stroke:#555,color:#fff,stroke-dasharray:5 5

    class REST,UI,TG,CRON client
    class OA,MEM,ROUTER,SYNTH,CONF,GUARD orchestrator
    class CA,PA,FA,WA,MA,EA,FBR agent
    class LMG,EMB,SM,LLM,WHISPER core
    class YF,FIN mcp
    class PC,LEDGER storage
    class WAL,MM,CHROME,GREEN,IMAP service
    class AGENTSJSON,MCPJSON,LS config
```

Property data is indexed into Pinecone offline by `agents.property.gatherers.pl_housing_data_collector` (CSV) and `agents.property.gatherers.data_collector` (URLs).
