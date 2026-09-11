"""
WAYFARER — AI Travel Agent (Streamlit frontend)

Wraps the LangGraph agent from app.ipynb (OpenAI + Tavily web search +
Duffel flights + Duffel stays + date/math tools) in a Streamlit chat UI
styled like a departures board.

Flow:
  1. Landing page asks only for an OpenAI API key.
  2. Once provided, the app moves to the chat interface.

Tavily and Duffel credentials are fixed in this file (not collected from
the user and never shown in the UI).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import uuid
from datetime import date, datetime
from typing import Annotated, Literal, Sequence, TypedDict

import requests
import streamlit as st

from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:  # pragma: no cover
    TavilySearchResults = None

# --------------------------------------------------------------------------
# FIXED CREDENTIALS — set once here, never asked from the user, never shown
# in the UI. Only the OpenAI key is collected from the person using the app.
# --------------------------------------------------------------------------
TAVILY_API_KEY = "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"
DUFFEL_ACCESS_TOKEN = "duffel_test_cvbF-ZiD1ozChHQW4vbZJBgn95mwU6KXVu5vvfGvkhy"

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

st.set_page_config(
    page_title="WAYFARER — AI Travel Agent",
    page_icon="✈️",
    layout="centered",
)

# --------------------------------------------------------------------------
# THEME (departures-board look, matching the original HTML mock)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

      :root{
        --navy: #0D1B2A;
        --navy-2: #142B41;
        --amber: #FFB627;
        --cream: #544c3f;
        --ink: #10151C;
        --slate: #7C8B9A;
        --green: #3CB371;
      }

      .stApp{ background: var(--cream); }
      html, body, [class*="css"]{ font-family:'Inter', sans-serif; color: var(--ink); }

      /* ---- header board ---- */
      .board{
        background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%);
        border-bottom: 6px solid var(--ink);
        border-radius: 6px;
        padding: 18px 22px 14px;
        margin-bottom: 18px;
      }
      .board-title{
        font-family:'Space Grotesk', sans-serif;
        font-weight:700;
        font-size: 26px;
        color: var(--cream);
        letter-spacing:0.02em;
      }
      .board-title span{ color: var(--amber); }
      .board-sub{
        font-family:'IBM Plex Mono', monospace;
        font-size: 12px;
        color: var(--slate);
        letter-spacing: 0.04em;
      }

      /* ---- quick-action board rows rendered as buttons ---- */
      div[data-testid="stButton"] > button{
        width:100%;
        background: var(--navy);
        color: var(--cream);
        border: 1px solid rgba(242,233,220,0.14);
        border-radius: 3px;
        font-family:'IBM Plex Mono', monospace;
        font-size: 12.5px;
        letter-spacing: 0.02em;
        text-align:left;
        padding: 10px 14px;
      }
      div[data-testid="stButton"] > button:hover{
        background: #1c3350;
        border-color: var(--amber);
        color: var(--amber);
      }

      /* ---- chat bubbles ---- */
      [data-testid="stChatMessage"]{
        border-radius: 3px;
        padding: 4px 2px;
      }

      /* ---- tool trace ---- */
      .tool-trace{
        font-family:'IBM Plex Mono', monospace;
        font-size: 11.5px;
        color: var(--slate);
        border-left: 2px solid var(--amber);
        padding: 4px 0 4px 10px;
        margin: 4px 0;
      }

      .foot-note{
        font-family:'IBM Plex Mono', monospace;
        font-size: 10.5px;
        color: var(--slate);
        text-align:center;
        margin-top: 18px;
        letter-spacing:0.03em;
      }

      /* ---- landing / gate page ---- */
      .gate-wrap{
        max-width: 420px;
        margin: 12vh auto 0;
        text-align:center;
      }
      .gate-plane{
        font-family:'Space Grotesk', sans-serif;
        font-size: 30px;
        margin-bottom: 6px;
      }
      .gate-sub{
        font-family:'IBM Plex Mono', monospace;
        font-size: 12.5px;
        color: var(--slate);
        margin-bottom: 22px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# TOOLS (ported from app.ipynb)
# --------------------------------------------------------------------------

DUFFEL_API_BASE = "https://api.duffel.com"

CITY_COORDINATES = {
    "NYC": {"latitude": 40.7128, "longitude": -74.0060},
    "PAR": {"latitude": 48.8566, "longitude": 2.3522},
    "LON": {"latitude": 51.5074, "longitude": -0.1278},
}


def _duffel_headers(version: str = "v2") -> dict:
    return {
        "Authorization": f"Bearer {DUFFEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Duffel-Version": version,
    }


@tool
def get_current_date_tool():
    """Returns the current date in 'YYYY-MM-DD' format. Useful for finding flights/hotels relative to today."""
    return date.today().isoformat()


@tool
def simple_math_tool(operand1: float, operand2: float, operation: Literal["add", "subtract"]):
    """
    Performs simple addition or subtraction on two numbers.
    Specify 'add' or 'subtract' for the operation.
    """
    if operation == "add":
        result = operand1 + operand2
        return f"The result of {operand1} + {operand2} is {result}"
    elif operation == "subtract":
        result = operand1 - operand2
        return f"The result of {operand1} - {operand2} is {result}"
    return "Invalid operation specified. Use 'add' or 'subtract'."


@tool
def search_flights_tool(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    travel_class: str = "ECONOMY",
    currency: str = "USD",
    max_offers: int = 5,
):
    """
    Searches live flight prices and availability using the Duffel Offer Requests API.

    Required:
        origin_code, destination_code - IATA airport/city codes (e.g., 'YYZ', 'CDG')
        departure_date - 'YYYY-MM-DD'

    Optional:
        return_date - for round trips; omit for one-way
        adults - number of adult passengers (default 1)
        travel_class - 'ECONOMY', 'PREMIUM_ECONOMY', 'BUSINESS', or 'FIRST'
        currency - 3-letter ISO currency code (default USD)
        max_offers - maximum number of offers to return
    """
    cabin_map = {
        "ECONOMY": "economy",
        "PREMIUM_ECONOMY": "premium_economy",
        "BUSINESS": "business",
        "FIRST": "first",
    }
    if travel_class.upper() not in cabin_map:
        return "Invalid travel_class. Use ECONOMY, PREMIUM_ECONOMY, BUSINESS, or FIRST."

    slices = [
        {
            "origin": origin_code.upper(),
            "destination": destination_code.upper(),
            "departure_date": departure_date,
        }
    ]
    if return_date:
        slices.append(
            {
                "origin": destination_code.upper(),
                "destination": origin_code.upper(),
                "departure_date": return_date,
            }
        )

    passengers = [{"type": "adult"} for _ in range(max(1, adults))]
    payload = {
        "data": {
            "cabin_class": cabin_map[travel_class.upper()],
            "slices": slices,
            "passengers": passengers,
        }
    }

    try:
        response = requests.post(
            f"{DUFFEL_API_BASE}/air/offer_requests",
            headers=_duffel_headers("v2"),
            json=payload,
            params={"return_offers": "true", "view": "offers"},
            timeout=60,
        )
        response.raise_for_status()
        response_data = response.json().get("data", {})
    except requests.RequestException as exc:
        return f"Duffel API request failed: {exc}"
    except ValueError:
        return "Duffel API returned an invalid JSON response."

    offers = response_data.get("offers", [])
    if not offers:
        return (
            f"No flight offers found for {origin_code} -> {destination_code} on {departure_date}"
            f"{' (return ' + return_date + ')' if return_date else ''}."
        )

    results = []
    for offer in offers[:max_offers]:
        total_amount = offer.get("total_amount", "N/A")
        total_currency = offer.get("total_currency", currency)
        itinerary_parts = []
        for flight_slice in offer.get("slices", []):
            segments = flight_slice.get("segments", [])
            if not segments:
                continue
            dep = segments[0].get("departing_at", "N/A")
            arr = segments[-1].get("arriving_at", "N/A")
            carriers = []
            for segment in segments:
                name = segment.get("operating_carrier", {}).get("name")
                if name and name not in carriers:
                    carriers.append(name)
            carrier_text = ", ".join(carriers) or "Unknown airline"
            itinerary_parts.append(f"{carrier_text} | {dep} -> {arr} | {len(segments) - 1} stop(s)")
        results.append(f"{' | '.join(itinerary_parts)} | {total_amount} {total_currency} | Offer: {offer.get('id', 'N/A')}")

    return "Found flight options:\n- " + "\n- ".join(results)


@tool
def search_hotels_tool(city_code: str, check_in_date: str, check_out_date: str, adults: int = 1):
    """
    Searches for available hotel options in a specific city for given dates
    using the Duffel Stays API.
    Requires a supported city code (e.g., 'NYC', 'PAR', 'LON') and dates in
    'YYYY-MM-DD' format. Use get_current_date_tool first if dates are relative.
    """
    coords = CITY_COORDINATES.get(city_code.upper())
    if not coords:
        return f"Unsupported city code '{city_code}'. Supported codes: {', '.join(CITY_COORDINATES.keys())}."

    payload = {
        "data": {
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "rooms": 1,
            "guests": [{"type": "adult"} for _ in range(max(1, adults))],
            "location": {"radius": 5, "geographic_coordinates": coords},
        }
    }

    try:
        response = requests.post(
            f"{DUFFEL_API_BASE}/stays/search",
            headers=_duffel_headers("v1"),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        results_data = response.json().get("data", {}).get("results", [])
    except requests.RequestException as exc:
        return f"Duffel Stays API request failed: {exc}"
    except ValueError:
        return "Duffel Stays API returned an invalid JSON response."

    if not results_data:
        return f"No hotels found for city code {city_code} on those dates."

    results = []
    for result in results_data[:5]:
        accommodation = result.get("accommodation", {})
        name = accommodation.get("name", "N/A")
        price = result.get("cheapest_rate_total_amount", "N/A")
        currency = result.get("cheapest_rate_currency", "")
        results.append(f"Hotel: {name}, Price: {price} {currency} (approx)")

    return "Found hotel options:\n- " + "\n- ".join(results)


def build_tools():
    tools = [get_current_date_tool, simple_math_tool, search_flights_tool, search_hotels_tool]
    if TavilySearchResults is not None:
        tools.append(TavilySearchResults(max_results=3))
    return tools


# --------------------------------------------------------------------------
# LANGGRAPH AGENT (ported from app.ipynb)
# --------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda a, b: list(a) + list(b)]


def build_agent_graph(tools):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    def call_model_with_tools(state: AgentState):
        model_with_tools = llm.bind_tools(tools)
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["action", "__end__"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "action"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model_with_tools)
    graph.add_node("action", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    return graph.compile()


# --------------------------------------------------------------------------
# PAGE 1 — GATE (collects only the OpenAI API key)
# --------------------------------------------------------------------------

def render_gate():
    st.markdown(
        """
        <div class="gate-wrap">
          <div class="gate-plane">✈ WAYFARER</div>
          <div class="gate-sub">ENTER YOUR OPENAI API KEY TO BOARD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("gate_form"):
            key_input = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
            submitted = st.form_submit_button("Continue →", use_container_width=True)

        if submitted:
            if not key_input.strip():
                st.error("Please enter an OpenAI API key to continue.")
            else:
                os.environ["OPENAI_API_KEY"] = key_input.strip()
                st.session_state.openai_key_set = True
                st.rerun()


# --------------------------------------------------------------------------
# PAGE 2 — CHAT INTERFACE
# --------------------------------------------------------------------------

def render_chat():
    st.markdown(
        f"""
        <div class="board">
          <div class="board-title">WAYFARER <span>/ DEPARTURES</span></div>
          <div class="board-sub">LOCAL TIME {datetime.now().strftime('%H:%M')} &nbsp;·&nbsp; TAP A ROW OR TYPE YOUR OWN REQUEST BELOW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🛫 Tools online")
        st.markdown("- 🟢 OpenAI (LLM)")
        st.markdown("- 🟢 Tavily (web search)")
        st.markdown("- 🟢 Duffel (flights & stays)")
        st.divider()
        if st.button("🧹 Clear conversation"):
            st.session_state.pop("lc_messages", None)
            st.session_state.pop("display_log", None)
            st.rerun()
        if st.button("🔒 Log out"):
            st.session_state.clear()
            os.environ.pop("OPENAI_API_KEY", None)
            st.rerun()

    QUICK_ACTIONS = [
        ("SUM · G1", "Summarize this in one paragraph, then translate it to Spanish: Electric cars use a battery pack to power an electric motor instead of an internal combustion engine."),
        ("WEB · G2", "What's the latest travel news for Paris right now?"),
        ("FLT · G3", "Find flights from Toronto (YYZ) to Paris (CDG) for 2 adults, departing June 1st."),
        ("STAY · G4", "Find a hotel in NYC from 2025-10-01 to 2025-10-03."),
        ("CALC · G5", "What is 128.50 plus 46.25?"),
    ]

    cols = st.columns(len(QUICK_ACTIONS))
    queued_prompt = None
    for col, (label, prompt) in zip(cols, QUICK_ACTIONS):
        with col:
            if st.button(label, key=f"qa_{label}"):
                queued_prompt = prompt

    if "lc_messages" not in st.session_state:
        st.session_state.lc_messages = []
    if "display_log" not in st.session_state:
        st.session_state.display_log = []

    if not st.session_state.display_log:
        st.info("✈️ Where to? Ask about flights, hotels, or today's travel news — or tap a row above.")

    for role, content, trace in st.session_state.display_log:
        with st.chat_message(role):
            if trace:
                with st.expander(f"🔧 {trace['name']}", expanded=False):
                    st.markdown(f"<div class='tool-trace'>{trace['content']}</div>", unsafe_allow_html=True)
            st.markdown(content)

    user_prompt = st.chat_input("I want to fly from Toronto to Paris in June…")
    prompt_to_run = queued_prompt or user_prompt

    if prompt_to_run:
        st.session_state.display_log.append(("user", prompt_to_run, None))
        st.session_state.lc_messages.append(HumanMessage(content=prompt_to_run))

        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant"):
            status = st.status("Routing request…", expanded=False)
            tools = build_tools()
            app = build_agent_graph(tools)

            final_text = ""
            try:
                result = app.invoke(
                    {"messages": st.session_state.lc_messages},
                    config={"recursion_limit": 15, "configurable": {"thread_id": str(uuid.uuid4())}},
                )
                new_messages = result["messages"][len(st.session_state.lc_messages):]

                for msg in new_messages:
                    if isinstance(msg, ToolMessage):
                        status.update(label=f"Used tool: {msg.name}", state="running")
                        st.session_state.display_log.append(
                            ("assistant", f"_Used **{msg.name}**_", {"name": msg.name, "content": msg.content})
                        )
                    elif isinstance(msg, AIMessage) and msg.content:
                        final_text = msg.content

                st.session_state.lc_messages = list(result["messages"])
                status.update(label="Done", state="complete")
            except Exception as exc:  # surface API/auth errors instead of a raw traceback
                final_text = f"Something went wrong calling the agent: {exc}"
                status.update(label="Error", state="error")

            st.markdown(final_text or "_(no response)_")
            st.session_state.display_log.append(("assistant", final_text or "_(no response)_", None))

        st.rerun()

    st.markdown(
        '<div class="foot-note">LIVE AGENT — OPENAI KEY FROM YOU · TAVILY & DUFFEL PRECONFIGURED</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------

if "openai_key_set" not in st.session_state:
    st.session_state.openai_key_set = False

if not st.session_state.openai_key_set:
    render_gate()
else:
    render_chat()
