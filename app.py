"""
app.py — Streamlit UI.

Usage:
    streamlit run app.py
"""
import streamlit as st
from llm import answer

st.set_page_config(page_title="CompliGuard", page_icon="⚖️", layout="wide")

st.title("CompliGuard")
st.caption("Interrogez la réglementation")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")
    model = st.text_input("Modèle Ollama", value="mistral")
    top_k = st.slider("Chunks à récupérer", min_value=1, max_value=10, value=5)
    livre_options = ["(tous)", "LIVRE I", "LIVRE II", "LIVRE III", "LIVRE IV", "LIVRE V"]
    livre_sel = st.selectbox("Filtrer par LIVRE", livre_options)
    livre = None if livre_sel == "(tous)" else livre_sel

# ── Chat state ────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "result": dict}

# ── Input ────────────────────────────────────────────────────────────────────
question = st.chat_input("Posez votre question en français ou en arabe…")

if question:
    with st.spinner("Recherche en cours…"):
        try:
            # Build flat message history for Ollama
            ollama_history = []
            for entry in st.session_state.history:
                ollama_history.append({"role": "user", "content": entry["question"]})
                ollama_history.append({"role": "assistant", "content": entry["result"]["answer"]})

            result = answer(question, model=model, top_k=top_k, livre=livre, history=ollama_history)
            st.session_state.history.append({"question": question, "result": result})
        except RuntimeError as e:
            st.error(str(e))

# ── Render history ────────────────────────────────────────────────────────────
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant"):
        st.markdown(entry["result"]["answer"])

        if entry["result"]["citations"]:
            with st.expander("Sources"):
                st.text(entry["result"]["citations"])

        if entry["result"]["chunks"]:
            with st.expander("Extraits récupérés"):
                for i, c in enumerate(entry["result"]["chunks"], 1):
                    location = " > ".join(
                        filter(None, [c.livre, c.document, c.titre, c.chapitre, c.article_ref])
                    )
                    st.markdown(f"**[{i}] {location}** (p.{c.page})")
                    st.text(c.text[:400] + ("…" if len(c.text) > 400 else ""))
                    st.divider()
