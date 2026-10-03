"""Hackathon starter: AI chat with Streamlit + Claude.

Run locally:  uv run streamlit run app.py
"""

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
SYSTEM_PROMPT = "You are a helpful assistant in a hackathon demo. Answer clearly and concisely."

# Server-side refusal fallback (Claude API): if the model declines, the API retries on a
# suitable fallback model within the same call. Only these models accept the "default" form.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

st.set_page_config(page_title="Hackathon Demo", page_icon="🚀")
st.title("🚀 Hackathon Demo")
st.caption(f"Model: {MODEL}")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning(
        "No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env`, add your key, "
        "and restart the app."
    )
    st.stop()

client = anthropic.Anthropic()

# `chat` holds plain text for display; `api_messages` holds the full content blocks
# returned by the API, which must be sent back unchanged on the next turn.
if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.api_messages = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.session_state.api_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    fallback_args = (
        {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        if MODEL in FALLBACK_MODELS
        else {}
    )

    with st.chat_message("assistant"):
        try:
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=64000,
                system=SYSTEM_PROMPT,
                messages=st.session_state.api_messages,
                **fallback_args,
            ) as stream:
                answer = st.write_stream(stream.text_stream)
                final = stream.get_final_message()
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Check ANTHROPIC_API_KEY in .env.")
            st.stop()
        except anthropic.NotFoundError:
            st.error(f"Model '{MODEL}' not found. Check ANTHROPIC_MODEL in .env.")
            st.stop()
        except anthropic.RateLimitError:
            st.error("Rate limit reached. Wait a moment and try again.")
            st.stop()
        except anthropic.APIStatusError as e:
            st.error(f"API error {e.status_code}: {e.message}")
            st.stop()
        except anthropic.APIConnectionError:
            st.error("Cannot reach the API. Check your internet connection.")
            st.stop()

        if final.stop_reason == "refusal":
            st.info("The model declined this request. Try rephrasing it.")

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.api_messages.append({"role": "assistant", "content": final.content})
