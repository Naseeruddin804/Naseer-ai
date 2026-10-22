"""Azure OpenAI chatbot (Streamlit).

Run with:  streamlit run app.py
"""
import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

BASE_DIR = Path(__file__).resolve().parent

# override=True -> the values in .env always win over old Windows/system
# environment variables, so a stale key or endpoint can never be used by mistake.
load_dotenv(BASE_DIR / ".env", override=True)

st.set_page_config(page_title="Naseer AI", page_icon="🤖", layout="centered")


CUSTOM_CSS = """
<style>
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"] { display: none !important; visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(124, 92, 255, 0.22), transparent 38%),
        radial-gradient(circle at 88% 4%, rgba(0, 200, 255, 0.18), transparent 35%),
        radial-gradient(circle at 50% 100%, rgba(255, 105, 180, 0.14), transparent 40%),
        #0f1020;
    color: #ecebff;
}
.block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 6rem; }

/* Hero header */
.hero {
    display: flex; align-items: center; gap: 16px;
    padding: 18px 22px; border-radius: 22px;
    background: linear-gradient(135deg, rgba(124,92,255,.35), rgba(0,200,255,.18));
    border: 1px solid rgba(255,255,255,.14);
    box-shadow: 0 10px 40px rgba(98, 85, 220, .30);
    backdrop-filter: blur(10px);
}
.hero .logo {
    width: 54px; height: 54px; border-radius: 16px; font-size: 28px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #7c5cff, #00c8ff);
    box-shadow: 0 6px 20px rgba(124,92,255,.55);
}
.hero h1 {
    margin: 0; padding: 0; font-size: 2rem; font-weight: 800; letter-spacing: .3px;
    background: linear-gradient(90deg, #ffffff, #b9a8ff 55%, #7fe3ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 2px 0 0; color: #b8b6dd; font-size: .92rem; }
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #3ddc97; margin-right: 6px; box-shadow: 0 0 8px #3ddc97;
}

/* Welcome */
.welcome-title {
    text-align: center; margin: 46px 0 4px; font-size: 2.1rem; font-weight: 800;
    background: linear-gradient(90deg, #b9a8ff, #7fe3ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.welcome-sub { text-align: center; color: #a9a7d0; margin-bottom: 26px; }

/* Buttons (suggestions + New Chat) */
.stButton > button {
    width: 100%; border-radius: 16px; padding: 14px 10px; font-weight: 600;
    color: #ecebff; background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.16);
    transition: all .2s ease;
}
.stButton > button:hover {
    transform: translateY(-3px); color: #fff;
    border-color: #8a6bff; background: rgba(124,92,255,.28);
    box-shadow: 0 8px 24px rgba(124,92,255,.35);
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 20px; padding: 14px 18px; margin: 10px 0;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.10);
    animation: pop .25s ease;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(124,92,255,.55), rgba(0,160,255,.35));
    border-color: rgba(160,140,255,.45);
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: #f2f1ff; line-height: 1.65; }
[data-testid="stChatMessage"] code { color: #ffd6f0; }
@keyframes pop { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* Input */
[data-testid="stChatInput"] {
    border-radius: 18px; border: 1px solid rgba(255,255,255,.18);
    background: rgba(20, 20, 45, .9); box-shadow: 0 6px 30px rgba(0,0,0,.35);
}
[data-testid="stChatInput"] textarea { color: #fff; }
[data-testid="stBottom"] > div { background: transparent; }

.footer-note { text-align: center; color: #8f8db8; font-size: .78rem; margin-top: 22px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SYSTEM_PROMPT = (
    "You are Naseer AI, a friendly and helpful AI assistant. "
    "Explain topics in simple language. "
    "Give examples when helpful."
)
MAX_MESSAGES_SENT = 20
MAX_MESSAGES_TOTAL = 40
MAX_INPUT_CHARS = 10000
SUGGESTIONS = [
    ("🧠 What is AI?", "Explain artificial intelligence in simple words."),
    ("🐍 Learn Python", "Teach me Python with a simple example."),
    ("📚 Study tips", "Give me three useful study tips."),
]


def clean(value):
    """Remove stray spaces/quotes that often sneak in when pasting a key."""
    return (value or "").strip().strip("\"'").strip()


def split_endpoint(endpoint):
    """Return 'https://host[/custom/path]' with any '/openai/...' part removed."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT must be a full URL, for example "
            "https://your-resource.openai.azure.com/openai/v1/"
        )
    path = parsed.path.rstrip("/")
    if "/openai" in path:
        path = path[: path.index("/openai")]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


@st.cache_resource(show_spinner=False)
def load_settings_and_client():
    """Read .env once and build the Azure client. Returns (settings, client)."""
    api_key = clean(os.getenv("AZURE_OPENAI_API_KEY"))
    endpoint = clean(os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment = clean(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    mode = clean(os.getenv("AZURE_OPENAI_API_MODE") or "v1").lower()
    api_version = clean(os.getenv("AZURE_OPENAI_API_VERSION")) or "2024-10-21"

    missing = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing settings in .env: " + ", ".join(missing))

    root = split_endpoint(endpoint)

    if mode == "v1":
        client = OpenAI(
            api_key=api_key,
            base_url=f"{root}/openai/v1/",
            timeout=60.0,
            max_retries=1,
        )
    elif mode == "standard":
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=root,
            api_version=api_version,
            timeout=60.0,
            max_retries=1,
        )
    else:
        raise ValueError("AZURE_OPENAI_API_MODE must be 'v1' or 'standard'.")

    # Secret is masked: only the length and last 4 characters are shown.
    print(
        f"[Azure config] mode={mode} | endpoint={root} | "
        f"deployment={deployment} | key=***{api_key[-4:]} ({len(api_key)} chars)"
    )
    return {"deployment": deployment}, client


def friendly_error(error, deployment):
    """Turn an Azure/OpenAI exception into a short message that says what to fix."""
    status = getattr(error, "status_code", None)
    code = str(getattr(error, "code", "") or "")
    name = type(error).__name__

    if code == "content_filter":
        return "Azure's content filter blocked this message. Try rephrasing it."
    if status == 401:
        return (
            "Azure rejected the API key (401). Check AZURE_OPENAI_API_KEY in .env "
            "- it must be a key from the SAME Azure resource as the endpoint."
        )
    if status == 403:
        return (
            "Access denied (403). The key has no permission for this resource, "
            "or the resource's network rules block your connection."
        )
    if status == 404:
        return (
            f"Azure could not find deployment '{deployment}' (404). Check "
            "AZURE_OPENAI_DEPLOYMENT (it must be the deployment NAME shown in "
            "Azure, not the model name) and AZURE_OPENAI_ENDPOINT."
        )
    if status == 429:
        return "Azure rate limit or quota reached (429). Wait a minute and try again."
    if "Timeout" in name:
        return "Azure took too long to answer. Please try again."
    if "Connection" in name:
        return (
            "Could not reach Azure. Check your internet connection and that "
            "AZURE_OPENAI_ENDPOINT is correct."
        )
    return f"Azure request failed ({name}: {str(error)[:300]})"


def ask_azure(client, deployment, history):
    """Send the recent conversation to Azure and return the reply text."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-MAX_MESSAGES_SENT:],
        ],
        max_completion_tokens=1000,
    )
    reply = response.choices[0].message.content if response.choices else None
    return reply or "The model returned an empty response."


# ---------------------------------------------------------------------------
# Load config (show a readable error in the page instead of crashing)
# ---------------------------------------------------------------------------
try:
    settings, client = load_settings_and_client()
except ValueError as config_error:
    st.error(str(config_error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
title_col, button_col = st.columns([5, 1], vertical_alignment="center")
title_col.markdown(
    """
    <div class="hero">
        <div class="logo">🤖</div>
        <div>
            <h1>Naseer AI</h1>
            <p><span class="status-dot"></span>Online · Your smart assistant</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if button_col.button("➕ New Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# Welcome screen + suggestions (only when the chat is empty)
# ---------------------------------------------------------------------------
pending_prompt = None

if not st.session_state.messages:
    st.markdown('<div class="welcome-title">How can I help you today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Ask me anything — I\'m Naseer AI, ready to help.</div>', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTIONS))
    for col, (label, prompt) in zip(cols, SUGGESTIONS):
        if col.button(label, use_container_width=True):
            pending_prompt = prompt

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# New input
# ---------------------------------------------------------------------------
typed = st.chat_input("Type your message...", max_chars=MAX_INPUT_CHARS)
user_text = (pending_prompt or typed or "").strip()

if user_text:
    if len(st.session_state.messages) >= MAX_MESSAGES_TOTAL:
        st.warning("Start a new chat to continue.")
    else:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_text)

        history = st.session_state.messages + [{"role": "user", "content": user_text}]

        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Thinking..."):
                    reply = ask_azure(client, settings["deployment"], history)
                st.markdown(reply)
                st.session_state.messages = history + [
                    {"role": "assistant", "content": reply}
                ]
            except Exception as error:
                # Full technical details appear in the terminal
                import logging

                logging.getLogger(__name__).exception("Azure API request failed")
                st.error(friendly_error(error, settings["deployment"]))
                # Failed message is not saved, so the history stays clean.

st.markdown('<div class="footer-note">Naseer AI can make mistakes. Verify important information.</div>', unsafe_allow_html=True)
