import uuid
import gradio as gr
from llm import answer

LIVRES = ["(all)", "LIVRE I", "LIVRE II", "LIVRE III", "LIVRE IV", "LIVRE V", "LIVRE VI"]


def build_sources_html(chunks) -> str:
    if not chunks:
        return ""
    rows = []
    for i, c in enumerate(chunks, 1):
        location = " › ".join(filter(None, [c.livre, c.titre, c.chapitre, c.section, c.article_ref]))
        rows.append(
            f"<li style='margin-bottom:12px'>"
            f"<b>[{i}]</b> {location} <i>(p.{c.page})</i>"
            f"<br><span style='white-space:pre-wrap;word-break:break-word'>{c.text}</span>"
            f"</li><hr style='border-color:#333;margin:4px 0'>"
        )
    return "<ul style='padding-left:12px;line-height:1.7;list-style:none'>" + "".join(rows) + "</ul>"


def to_ollama_history(gradio_history: list) -> list:
    result = []
    for m in gradio_history:
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if m["role"] == "assistant" and "\n\n---\n**Sources**" in content:
            content = content.split("\n\n---\n**Sources**")[0]
        result.append({"role": m["role"], "content": content})
    return result


def make_conv_id() -> str:
    return str(uuid.uuid4())


def conv_title(history: list) -> str:
    for m in history:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, list):
                content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            return content[:40] + ("…" if len(content) > 40 else "")
    return "Nouvelle conversation"


def sidebar_choices(convs: dict) -> list:
    return [
        (conv_title(data["history"]) if data["history"] else "Nouvelle conversation", cid)
        for cid, data in convs.items()
    ]


def respond(message, display_history, convs, current_id, model, top_k, livre_sel):
    ollama_history = to_ollama_history(display_history)
    livre = None if livre_sel == "(all)" else livre_sel

    result = answer(message, model=model, top_k=int(top_k), livre=livre, history=ollama_history)

    response = result["answer"]
    if result["citations"]:
        response += "\n\n---\n**Sources**\n" + result["citations"]

    sources = build_sources_html(result["chunks"])

    new_history = display_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]

    convs[current_id]["history"] = new_history
    choices = sidebar_choices(convs)

    return "", new_history, convs, gr.update(choices=choices, value=current_id), sources


def new_conversation(convs):
    cid = make_conv_id()
    convs[cid] = {"history": []}
    choices = sidebar_choices(convs)
    return convs, cid, [], gr.update(choices=choices, value=cid), ""


def switch_conversation(selected_id, convs):
    if selected_id and selected_id in convs:
        return convs[selected_id]["history"], ""
    return [], ""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

body, .gradio-container, .main {
    background: #0c0c0f !important;
    font-family: 'Inter', sans-serif !important;
}

#title {
    text-align: center;
    color: #c9a84c !important;
    font-size: 2em !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}
#subtitle {
    text-align: center;
    color: #8a7a55 !important;
    font-size: 0.9em !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin-top: -4px !important;
}

*, p, span, label, .label-wrap span, h1, h2, h3, h4 { color: #e8e4dc !important; }

button.primary {
    background: linear-gradient(135deg, #c9a84c, #a8893a) !important;
    border-color: #c9a84c !important;
    color: #0c0c0f !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #d4b96a, #c9a84c) !important;
    border-color: #d4b96a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px #c9a84c40 !important;
}
button.secondary {
    background: #141418 !important;
    border: 1px solid #2a2830 !important;
    color: #c9a84c !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
button.secondary:hover {
    background: #1e1c24 !important;
    border-color: #c9a84c !important;
    color: #d4b96a !important;
}

#new-btn { width: 100%; }

input, textarea {
    background: #141418 !important;
    border: 1px solid #2a2830 !important;
    border-radius: 6px !important;
    color: #e8e4dc !important;
    transition: border-color 0.2s ease !important;
}
input::placeholder, textarea::placeholder { color: #55504a !important; }
input:focus, textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px #c9a84c25 !important;
    outline: none !important;
}

.chatbot, [data-testid="chatbot"] {
    background: #101013 !important;
    border: 1px solid #2a2830 !important;
    border-radius: 10px !important;
}
.prose, .prose * { color: #e8e4dc !important; line-height: 1.7 !important; }

#sources-box {
    overflow-y: auto;
    font-size: 0.84em;
    background: #101013;
    border: 1px solid #2a2830;
    border-radius: 10px;
    padding: 12px;
    color: #e8e4dc;
    line-height: 1.6;
}
#sources-box * { color: #e8e4dc !important; }
#sources-box b { color: #c9a84c !important; }
#sources-box i { color: #8a7a55 !important; }
#sources-box hr { border-color: #2a2830 !important; margin: 6px 0 !important; }

.block, .panel {
    background: #0c0c0f !important;
    border-color: #1e1c24 !important;
    border-radius: 10px !important;
}
input[type="radio"] { accent-color: #c9a84c !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0c0c0f; }
::-webkit-scrollbar-thumb { background: #2a2830; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c9a84c; }

footer { display: none !important; }
"""

initial_id = make_conv_id()
initial_convs = {initial_id: {"history": []}}

with gr.Blocks(title="CompliGuard-FR") as demo:

    convs_state = gr.State(initial_convs)
    current_id_state = gr.State(initial_id)

    gr.Markdown("# ⚖️ CompliGuard-FR", elem_id="title")
    gr.Markdown("AMF regulatory assistant — French financial markets regulation", elem_id="subtitle")

    with gr.Row():

        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### Conversations")
            new_btn = gr.Button("＋ Nouvelle conversation", variant="primary", elem_id="new-btn")
            conv_selector = gr.Radio(
                choices=sidebar_choices(initial_convs),
                value=initial_id,
                label="",
                show_label=False,
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Posez votre question en français / Ask in English…",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Envoyer", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Paramètres")
            model_input = gr.Textbox(label="Modèle (claude / ollama)", value="claude")
            top_k_slider = gr.Slider(1, 15, value=8, step=1, label="Chunks récupérés")
            livre_dd = gr.Dropdown(LIVRES, value="(all)", label="Filter by LIVRE")
            gr.Markdown("### Extraits récupérés")
            sources_html = gr.HTML(elem_id="sources-box")

    respond_inputs = [msg_box, chatbot, convs_state, current_id_state, model_input, top_k_slider, livre_dd]
    respond_outputs = [msg_box, chatbot, convs_state, conv_selector, sources_html]

    msg_box.submit(respond, respond_inputs, respond_outputs)
    send_btn.click(respond, respond_inputs, respond_outputs)

    new_btn.click(
        new_conversation,
        inputs=[convs_state],
        outputs=[convs_state, current_id_state, chatbot, conv_selector, sources_html],
    )

    conv_selector.change(
        switch_conversation,
        inputs=[conv_selector, convs_state],
        outputs=[chatbot, sources_html],
    ).then(
        lambda sid: sid,
        inputs=[conv_selector],
        outputs=[current_id_state],
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        theme=gr.themes.Base(primary_hue="yellow", neutral_hue="neutral"),
        css=CSS,
    )
