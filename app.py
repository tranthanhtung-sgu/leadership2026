import streamlit as st
from openai import OpenAI
import time
import random
import os
import pandas as pd
from datetime import datetime

from character_configs import (
    AI_NAMES,
    CHARACTERS,
    format_typing_delay,
    parse_typing_delay,
    pick_next_speaker,
)
from agents import get_agent

# ==========================================
# 0. PASSWORD PROTECTION
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if not st.session_state.password_correct:
        pw = st.text_input("Enter Access Code", type="password")
        if pw == "sonny@curtin":
            st.session_state.password_correct = True
            st.rerun()
        else:
            if pw: st.error("Wrong password")
            st.stop()

check_password()

# ==========================================
# 1. INITIAL SETUP & API
# ==========================================
st.set_page_config(page_title="Leadership Study v3", layout="wide")

if "OPENAI_KEY" not in st.secrets:
    st.error("Please add OPENAI_KEY to your Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
MODEL_NAME = "gpt-5.2"

# Timing tuned for snappier chat (seconds)
FRAGMENT_REFRESH_SEC = 1.2
START_WARMUP_SEC = 0.8
HUMAN_REPLY_DELAY_RANGE = (1.5, 3.2)
IDLE_AFTER_BURST_RANGE = (5.0, 9.0)

# ==========================================
# 2. DATA LOADING FUNCTIONS
# ==========================================
def load_text_file(filename, default_text):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return default_text

initial_scenario = load_text_file("scenario.txt", "Scenario: Team meeting about CortiSense.")
initial_behavioral = load_text_file("behavioural_requirements.txt", "Rules: Act like humans, use typos.")

# ==========================================
# 3. SESSION STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sim_active" not in st.session_state:
    st.session_state.sim_active = False
if "next_ai_time" not in st.session_state:
    st.session_state.next_ai_time = 0.0
if "ai_burst_remaining" not in st.session_state:
    st.session_state.ai_burst_remaining = 0
if "api_count" not in st.session_state:
    st.session_state.api_count = 0

for _name in AI_NAMES:
    _wk = f"tune_w_{_name}"
    _tk = f"tune_td_{_name}"
    if _wk not in st.session_state:
        st.session_state[_wk] = float(CHARACTERS[_name].default_weight)
    if _tk not in st.session_state:
        st.session_state[_tk] = format_typing_delay(CHARACTERS[_name].typing_delay)

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🔬 Research Controller")

st.sidebar.metric("API Requests (Session)", st.session_state.api_count)

with st.sidebar.expander("🛠️ API Diagnostic Tool", expanded=False):
    if st.button("Verify Connection"):
        try:
            test_resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5
            )
            st.success("API Response Received!")
            st.session_state.api_count += 1
        except Exception as e:
            st.error(f"Connection Failed: {e}")

participant_id = st.sidebar.text_input("Participant ID", value="P-001")
leadership_style = st.sidebar.select_slider(
    "Zoe Leadership Style",
    options=["Non-Inclusive", "Inclusive"]
)

with st.sidebar.expander("🎛️ Character pacing (live)", expanded=True):
    st.caption(
        "Weights: baseline chance each teammate speaks. "
        "If one or more teammates are directly asked (@name, 'Name, …', or name before ?), "
        "one respondent is picked using these weights; low weight = less often 'takes' a solo ask."
    )
    for _n in AI_NAMES:
        st.slider(
            f"{_n} — speak weight",
            min_value=0.05,
            max_value=1.5,
            step=0.05,
            key=f"tune_w_{_n}",
        )
        st.text_input(
            f"{_n} — typing delay (min,max sec)",
            key=f"tune_td_{_n}",
            help="Example: 0.15,0.45",
        )

with st.sidebar.expander("📝 Edit Scenario / Rules", expanded=False):
    scenario_content = st.text_area("Scenario Context", value=initial_scenario, height=150)
    behavioral_content = st.text_area("Behavioral Rules", value=initial_behavioral, height=150)

st.sidebar.divider()
col1, col2 = st.sidebar.columns(2)
if col1.button("▶ START SIM"):
    st.session_state.sim_active = True
    st.session_state.next_ai_time = time.time() + START_WARMUP_SEC
    st.session_state.ai_burst_remaining = random.randint(2, 3)
    st.rerun()
if col2.button("⏹ RESET"):
    st.session_state.sim_active = False
    st.session_state.messages = []
    st.session_state.next_ai_time = 0.0
    st.session_state.ai_burst_remaining = 0
    st.session_state.api_count = 0
    st.rerun()

if st.sidebar.button("💾 EXPORT TRANSCRIPT"):
    if st.session_state.messages:
        df = pd.DataFrame(st.session_state.messages)
        df['participant_id'] = participant_id
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label="Download CSV", data=csv, file_name=f"study_{participant_id}.csv", mime="text/csv")

# ==========================================
# 5. AI ENGINE (three distinct agents; one API call per turn)
# ==========================================
def run_agent_turn(speaker: str):
    agent = get_agent(speaker)
    cfg = agent.config
    try:
        txt, n_api = agent.generate_reply(
            client,
            MODEL_NAME,
            st.session_state.messages,
            leadership_style,
            scenario_content,
            behavioral_content,
        )
        st.session_state.messages.append({
            "speaker": speaker, "text": txt, "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.api_count += n_api
    except Exception as e:
        st.error(f"AI Error ({speaker}): {e}")
    return cfg

# ==========================================
# 6. MAIN UI - INFO BAR
# ==========================================
st.info(f"Condition: **{leadership_style}** | Participant: **{participant_id}**")

# ==========================================
# 7. ASYNC CHAT UI
# ==========================================
@st.fragment(run_every=FRAGMENT_REFRESH_SEC)
def chat_ui():
    for m in st.session_state.messages:
        with st.chat_message(m["speaker"]):
            st.write(f"**{m['speaker']}**:")
            st.text(m['text'])

    if not st.session_state.sim_active:
        st.caption("Simulation not started. Use ▶ START SIM in the sidebar.")
        return

    now = time.time()
    if now >= st.session_state.next_ai_time:
        if not st.session_state.messages:
            last_speaker, last_text = "Participant", ""
        else:
            last = st.session_state.messages[-1]
            last_speaker, last_text = last["speaker"], last.get("text", "")

        weight_map = {n: float(st.session_state[f"tune_w_{n}"]) for n in AI_NAMES}
        speaker, cfg = pick_next_speaker(
            last_speaker,
            last_text,
            weights_override=weight_map,
            recent_messages=st.session_state.messages,
        )

        td_raw = st.session_state.get(
            f"tune_td_{speaker}",
            format_typing_delay(cfg.typing_delay),
        )
        lo, hi = parse_typing_delay(str(td_raw), cfg.typing_delay)

        with st.spinner(f"{speaker} is typing..."):
            time.sleep(random.uniform(lo, hi))
            cfg = run_agent_turn(speaker)

        g0, g1 = cfg.burst_gap
        gap_after = random.uniform(g0, g1)

        if st.session_state.ai_burst_remaining > 0:
            st.session_state.next_ai_time = time.time() + gap_after
            st.session_state.ai_burst_remaining -= 1
        else:
            st.session_state.next_ai_time = time.time() + random.uniform(*IDLE_AFTER_BURST_RANGE)
            st.session_state.ai_burst_remaining = random.randint(1, 3)

    if prompt := st.chat_input("Join the conversation... (type any time)"):
        st.session_state.messages.append({
            "speaker": "Participant",
            "text": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.next_ai_time = time.time() + random.uniform(*HUMAN_REPLY_DELAY_RANGE)
        # Keep it snappy but avoid 2 AI messages firing immediately after the human;
        # that often causes repeated/incomplete re-introductions.
        st.session_state.ai_burst_remaining = 1
        st.rerun()

chat_ui()
