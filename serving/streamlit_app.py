import streamlit as st

from core.config import GENERATION_MODEL
from core.generation.llm_client import GeminiClient
from corpus.model_facts import MODELS
from domains.instrument_support.agent import InstrumentSupportAgent, Ticket
from domains.instrument_support.config import KNOWN_MODEL_NUMBERS, build_retriever
from observability.db import log_request


@st.cache_resource
def get_agent() -> InstrumentSupportAgent:
    client = GeminiClient()
    retriever = build_retriever()
    return InstrumentSupportAgent(retriever, client, GENERATION_MODEL, KNOWN_MODEL_NUMBERS)


st.title("Instrument Technical Support Assistant")
st.caption(
    "Demo assistant over a synthetic, LLM-generated instrument documentation corpus "
    "for a fictional line of lab instruments. See the README for details."
)

model_numbers = ["(unknown)"] + sorted(m["model_number"] for m in MODELS)
model_number = st.selectbox("Instrument model", model_numbers)
symptom = st.text_input("Symptom or error code", placeholder="e.g. E-104")
free_text = st.text_area("Additional details (optional)")

if st.button("Get resolution") and symptom:
    agent = get_agent()
    ticket = Ticket(
        symptom_or_error_code=symptom,
        model_number=None if model_number == "(unknown)" else model_number,
        free_text=free_text,
    )
    response, metadata = agent.handle_with_metadata(ticket)
    log_request(symptom, response, metadata)

    if response.escalate:
        st.warning("Low confidence — this ticket should be escalated to a human technician.")

    st.write(response.draft)
    st.metric("Confidence", f"{response.confidence:.2f}")

    if response.citations:
        with st.expander("Cited sources"):
            for citation in response.citations:
                st.write(f"- {citation}")
