# Robust Finance Research System (LangGraph + Tavily)

Stateful multi-agent workflow for SET50 research that fuses real-time web search, technical indicators, and bilingual reporting (Thai with English technical terms). Built with LangGraph.

## Features
- Cyclic StateGraph: Researcher → Analyst → Writer → Critic, with bounded revision loop.
- Data sources: yfinance OHLCV (1-month) + Tavily news search.
- Analytics: RSI(14), MACD + signal, SMA20/50 trend signal.
- Output: Thai narrative with English technical terms; critic enforces structure/consistency/news grounding.

## Quickstart
1) Python 3.10+ recommended.  
2) Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3) Set environment keys (edit `.env`):
   ```
   OPENAI_API_KEY=YOUR_OPENAI_API_KEY
   TAVILY_API_KEY=YOUR_TAVILY_API_KEY
   ```
4) Run a sample workflow:
   ```bash
   python -c "from graph import run_workflow; print(run_workflow('PTT.BK'))"
   ```

## Files
- `state.py` — Shared `AgentState` TypedDict.
- `tools.py` — yfinance fetcher + Tavily search wrapper (graceful fallbacks, .env loading).
- `nodes.py` — Researcher, Analyst, Writer, Critic node functions.
- `graph.py` — Builds/compiles the LangGraph and exposes `run_workflow`.
- `.env` — Place your API keys (excluded from git).
- `.gitignore` — Ignores venvs, bytecode, `.env`.

## Notes
- Critic loop stops after `max_revisions` (default 2) to avoid infinite retries.
- News search is best-effort; if Tavily fails or key missing, the flow continues with a placeholder.
- Writer always cites at least one provided news line when available. Use SET tickers with `.BK` (e.g., `PTT.BK`, `ADVANC.BK`).
