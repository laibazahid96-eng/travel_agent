# WAYFARER — Streamlit frontend

A Streamlit chat UI for the LangGraph travel agent from `app.ipynb`: OpenAI for
reasoning, Tavily for live web search, and Duffel for real flight/hotel search,
with a "departures board" visual theme.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tavily and Duffel credentials are **hardcoded directly in `app.py`** — the
app never asks for them and never displays them in the UI. Only the OpenAI
key is collected, on a landing "gate" page, before the chat interface loads.

⚠️ Because the Tavily/Duffel keys live in the source file, treat `app.py`
itself as a secret — don't commit it to a public repo or share it as-is.

## What it does

- **OpenAI (`gpt-4.1-mini`)** — the core reasoning model, with tools bound via
  LangGraph's `bind_tools`.
- **Tavily** — live web search (news, advisories), used only if a Tavily key
  is present.
- **Duffel** — real flight search (`/air/offer_requests`) and hotel search
  (`/stays/search`, limited to NYC/PAR/LON via a coordinate lookup, same as
  the notebook).
- **Date & math tools** — small utility tools ported directly from the
  notebook.

The graph structure mirrors `build_graph_one_tool` in the notebook: an
`agent` node that calls the LLM with tools bound, a conditional edge that
routes to an `action` (`ToolNode`) step whenever the model requests a tool
call, and a loop back to `agent` until the model returns a plain answer.

## Notes

- Tool calls are shown in the chat as collapsible "🔧 tool name" traces so you
  can see what the agent actually did.
- The quick-action board (SUM / WEB / FLT / STAY / CALC) fires the same
  canned prompts as the original HTML mockup, but now runs the real agent
  instead of a simulated response.
- "Log out" in the sidebar clears the session and returns to the gate page,
  in case someone else uses the same machine.
