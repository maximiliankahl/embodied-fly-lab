"""Embodied Fly Lab - Streamlit dashboard.

Run locally:   uv run streamlit run app.py
Replay mode:   set FLYLAB_REPLAY=1 (recorded runs + benchmark files only, no live simulation)

Pages live in flylab/ui_*.py and are imported lazily, so the app starts fast and also
renders without the simulation stack (flygym, connectome files).
"""

import streamlit as st

st.set_page_config(page_title="Embodied Fly Lab", layout="wide", initial_sidebar_state="expanded")

from flylab import ui  # noqa: E402  (light: stdlib + streamlit only)


def _notebook() -> None:
    from flylab import ui_notebook

    ui_notebook.page()


def _bench() -> None:
    from flylab import ui_bench

    ui_bench.page()


def _validation() -> None:
    from flylab import ui_validation

    ui_validation.page()


def _method() -> None:
    from flylab import ui_method

    ui_method.page()


ui.inject_css()
nav = st.navigation(
    [
        st.Page(_notebook, title="Lab notebook", icon=":material/menu_book:", url_path="notebook", default=True),
        st.Page(_bench, title="Experiment bench", icon=":material/science:", url_path="bench"),
        st.Page(_validation, title="Validation & speed", icon=":material/fact_check:", url_path="validation"),
        st.Page(_method, title="Method & limits", icon=":material/account_tree:", url_path="method"),
    ]
)

with st.sidebar:
    st.markdown("### Embodied Fly Lab")
    st.caption(
        "An Omnigent agent lab that runs in-silico activation and silencing experiments on a whole-brain "
        "Drosophila connectome model (FlyWire v783) coupled to a NeuroMechFly physics body, and checks "
        "the results against published experiments."
    )
    kr = ui.key_results()
    if kr:
        st.markdown("**Key results**")
        for label, value in kr:
            st.markdown(f"**{value}** <span class='fl-small'>{label}</span>", unsafe_allow_html=True)
    replay, reasons = ui.replay_status()
    if replay:
        st.markdown(ui.badge("Replay mode", "orange") + " recorded data only")
        st.caption("; ".join(reasons))
    else:
        st.markdown(ui.badge("Live mode", "green") + " simulation stack available")
    st.caption("Hack-Nation 7 · Databricks challenge: Agentic Scientific Discovery")

nav.run()
