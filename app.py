import html
import io
import os
import random
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI

from character_configs import (
    AI_NAMES,
    CHARACTERS,
    format_typing_delay,
    parse_typing_delay,
    pick_next_speaker,
)

try:
    from character_configs import plan_contribution_nudge
except ImportError:  # deploy may have older character_configs without this helper

    def plan_contribution_nudge(*args, **kwargs):
        """No-op: quiet-teammate draw-ins disabled until character_configs is updated."""
        return None


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
# Longer pause after the human sends so they can read before bots pile on
HUMAN_REPLY_DELAY_RANGE = (7.0, 16.0)
IDLE_AFTER_BURST_RANGE = (5.0, 9.0)
# Extra idle after a burst if the Participant spoke recently (reading / jump-in window)
PARTICIPANT_RECENT_IDLE_RANGE = (16.0, 32.0)
# After this many consecutive AI lines without the Participant, Zoe gets a soft “human in the room” nudge.
# Repeats every PARTICIPANT_INVITE_REPEAT_EVERY further AI turns so a silent Participant isn’t dropped after the opening.
PARTICIPANT_INVITE_FIRST_BOT_TURNS = 5
PARTICIPANT_INVITE_REPEAT_EVERY = 7
PARTICIPANT_INVITE_EXTRA = (
    "(Facilitator nudge: the human teammate is in this chat but has been quiet — "
    "one short, natural line that leaves them room to jump in or reacts to the thread without demanding an answer; "
    "do not re-ask the same question or A/B you already used; you can mostly talk to teammates but nod to them.)"
)


def _participant_invite_due(bot_turns: int, first: int, every: int) -> bool:
    """True on first at `first`, then every `every` additional AI-only turns (e.g. 5, 12, 19, …)."""
    if bot_turns < first:
        return False
    if bot_turns == first:
        return True
    return (bot_turns - first) % every == 0


# Quiet-teammate draw-ins (sidebar controls removed; adjust here if needed.)
QUIET_DRAW_IN_LINES_BELOW = 3
QUIET_DRAW_IN_WINDOW = 40
QUIET_DRAW_IN_MIN_AI = 8
QUIET_DRAW_IN_COOLDOWN = 5
QUIET_DRAW_IN_ADDRESS_LOOKBACK = 10
QUIET_DRAW_IN_ROLL = 0.5
POST_HUMAN_BOT_WINDOW = 5

# Scroll this area only; keeps the participant chat box from being pushed off-screen.
CHAT_SCROLL_HEIGHT_PX = 560

# WhatsApp-Web–style peer row: pastel avatar, coloured display name
_SPEAKER_CHAT_STYLE: dict[str, dict[str, str]] = {
    "Zoe": {"avatar_bg": "#cfefff", "avatar_fg": "#075e54", "name": "#0284c7"},
    "Femke": {"avatar_bg": "#ffe8cc", "avatar_fg": "#a05000", "name": "#dd6b20"},
    "Hao": {"avatar_bg": "#d8f5e3", "avatar_fg": "#075e54", "name": "#128c7e"},
}
_DEFAULT_PEER_STYLE = {"avatar_bg": "#e1e8e4", "avatar_fg": "#54656f", "name": "#54656f"}


def _team_chat_message_html(messages: list[dict], participant_label: str, feed_max_px: int) -> str:
    """WhatsApp-like pane: group header, doodle-pattern feed, left incoming / right outgoing."""
    rows: list[str] = []
    for m in messages:
        speaker = str(m.get("speaker") or "?")
        raw = m.get("text") or ""
        text_safe = html.escape(raw).replace("\n", "<br/>")
        ts = html.escape(str(m.get("timestamp") or ""))
        if speaker == "Participant":
            tick = '<span class="tc-ticks" aria-hidden="true">✓✓</span>'
            time_html = f'<span class="tc-time">{ts}</span>' if ts else ""
            rows.append(
                '<div class="tc-row tc-row-self">'
                '<div class="tc-bubble-wrap tc-bubble-wrap-self">'
                '<div class="tc-bubble tc-bubble-self">'
                f'<div class="tc-bubble-body">{text_safe}</div>'
                f'<div class="tc-bubble-foot">{time_html}{tick}</div>'
                "</div></div></div>"
            )
        else:
            st = _SPEAKER_CHAT_STYLE.get(speaker, _DEFAULT_PEER_STYLE)
            initial = html.escape(speaker[:1].upper())
            nm = html.escape(speaker)
            time_html = f'<span class="tc-time">{ts}</span>' if ts else ""
            rows.append(
                '<div class="tc-row tc-row-peer">'
                f'<div class="tc-avatar" style="background:{st["avatar_bg"]};color:{st["avatar_fg"]};">'
                f"{initial}</div>"
                '<div class="tc-bubble-wrap">'
                f'<div class="tc-peer-name" style="color:{st["name"]};">{nm}</div>'
                '<div class="tc-bubble tc-bubble-peer">'
                f'<div class="tc-bubble-body">{text_safe}</div>'
                f'<div class="tc-bubble-foot">{time_html}</div>'
                "</div></div></div>"
            )
    body = "".join(rows)
    pl = html.escape(participant_label)
    names_sub = html.escape("Zoe, Femke, Hao, You")
    return f"""
<div class="tc-window" style="min-height:{CHAT_SCROLL_HEIGHT_PX}px; max-height:{CHAT_SCROLL_HEIGHT_PX}px;">
  <header class="tc-header">
    <div class="tc-header-lead">
      <div class="tc-header-icon" aria-hidden="true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="9" cy="8" r="3.5" stroke="currentColor" stroke-width="1.6"/>
          <circle cx="15" cy="9" r="2.8" stroke="currentColor" stroke-width="1.6"/>
          <path d="M4 19.5c.8-4 4.2-6 8-6s7.2 2 8 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="tc-header-title">Study team chat</div>
        <div class="tc-header-sub">{names_sub} · <span class="tc-header-id">{pl}</span></div>
      </div>
    </div>
  </header>
  <div class="tc-feed" style="min-height:{feed_max_px}px; height:{feed_max_px}px; max-height:{feed_max_px}px; box-sizing:border-box;">{body}</div>
</div>
"""


def _team_chat_css() -> str:
    return """
<style>
/* WhatsApp-Web–inspired team chat (not GPT single-column layout) */
.tc-window {
  display: flex;
  flex-direction: column;
  border-radius: 0;
  overflow: hidden;
  border: 1px solid #d1d7db;
  box-shadow: 0 1px 4px rgba(11,20,26,.08);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  flex: 1 1 auto;
}
.tc-header {
  flex-shrink: 0;
  padding: 0.5rem 1rem;
  background: #f0f2f5;
  border-bottom: 1px solid #e9edef;
}
.tc-header-lead {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}
.tc-header-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #25d366;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.tc-header-title {
  font-weight: 600;
  font-size: 1rem;
  color: #111b21;
  letter-spacing: -0.02em;
}
.tc-header-sub {
  font-size: 0.8125rem;
  color: #667781;
  margin-top: 0.05rem;
  line-height: 1.25;
}
.tc-header-id { color: #41525d; font-weight: 500; }
/* Feed: full-height WhatsApp-style beige + subtle texture (empty or not) */
.tc-feed {
  overflow-y: auto;
  overflow-x: hidden;
  padding: 0.6rem 0.65rem 1rem;
  flex: 1 1 auto;
  min-height: 0;
  background-color: #efeae2;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'%3E%3Cg fill='%23d1ccc4' fill-opacity='0.35'%3E%3Ccircle cx='8' cy='12' r='1.2'/%3E%3Ccircle cx='52' cy='38' r='0.9'/%3E%3Ccircle cx='28' cy='64' r='1'/%3E%3Ccircle cx='70' cy='8' r='0.8'/%3E%3Ccircle cx='38' cy='22' r='0.7'/%3E%3Cpath d='M60 55 L62 58 L59 58 Z'/%3E%3Crect x='20' y='45' width='3' height='3' rx='0.5' opacity='0.6'/%3E%3C/g%3E%3C/svg%3E");
}
.tc-row {
  display: flex;
  align-items: flex-end;
  gap: 0.4rem;
  margin-bottom: 0.35rem;
  max-width: 100%;
}
.tc-row-peer { justify-content: flex-start; }
.tc-row-self { justify-content: flex-end; }
.tc-avatar {
  width: 32px;
  height: 32px;
  min-width: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.8rem;
  align-self: flex-end;
  margin-bottom: 2px;
}
.tc-bubble-wrap { max-width: min(82%, 480px); display: flex; flex-direction: column; }
.tc-bubble-wrap-self { align-items: flex-end; }
.tc-peer-name {
  font-size: 0.8125rem;
  font-weight: 600;
  margin: 0 0 0.15rem 0.35rem;
}
.tc-bubble {
  position: relative;
  padding: 0.35rem 0.55rem 0.45rem 0.65rem;
  border-radius: 7.5px;
  font-size: 0.9025rem;
  line-height: 1.42;
  color: #111b21;
  word-wrap: break-word;
  box-shadow: 0 1px 0.5px rgba(11,20,26,.13);
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 0.2rem 0.45rem;
}
.tc-bubble-peer {
  background: #fff;
  border-top-left-radius: 0;
}
.tc-bubble-self {
  background: #d9fdd3;
  border-top-right-radius: 0;
}
.tc-bubble-body {
  flex: 1 1 auto;
  min-width: 3.5rem;
}
.tc-bubble-foot {
  display: inline-flex;
  align-items: center;
  gap: 0.15rem;
  flex: 0 0 auto;
  margin-left: auto;
}
.tc-time {
  font-size: 0.6875rem;
  color: #667781;
  white-space: nowrap;
  line-height: 1;
}
.tc-bubble-self .tc-time { color: #667781; }
.tc-ticks {
  font-size: 0.65rem;
  color: #53bdeb;
  letter-spacing: -2px;
  line-height: 1;
}
@media (prefers-color-scheme: dark) {
  .tc-window { border-color: #2a3942; }
  .tc-header { background: #202c33; border-bottom-color: #2a3942; }
  .tc-header-title { color: #e9edef; }
  .tc-header-sub { color: #8696a0; }
  .tc-header-id { color: #aebac1; }
  .tc-header-icon { background: #25d366; }
  .tc-feed {
    background-color: #0b141a;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'%3E%3Cg fill='%232a3942' fill-opacity='0.5'%3E%3Ccircle cx='8' cy='12' r='1.2'/%3E%3Ccircle cx='52' cy='38' r='0.9'/%3E%3Ccircle cx='28' cy='64' r='1'/%3E%3Ccircle cx='70' cy='8' r='0.8'/%3E%3C/g%3E%3C/svg%3E");
  }
  .tc-bubble-peer { background: #202c33; color: #e9edef; box-shadow: 0 1px 0.5px rgba(0,0,0,.35); }
  .tc-bubble-self { background: #005c4b; color: #e9edef; }
  .tc-time { color: #8696a0 !important; }
}
</style>
"""


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
if "bot_turns_since_human" not in st.session_state:
    st.session_state.bot_turns_since_human = 0
if "last_contribution_nudge_at_ai_count" not in st.session_state:
    st.session_state.last_contribution_nudge_at_ai_count = -10_000

for _name in AI_NAMES:
    _wk = f"tune_w_{_name}"
    _tk = f"tune_td_{_name}"
    if _wk not in st.session_state:
        st.session_state[_wk] = float(CHARACTERS[_name].default_weight)
    if _tk not in st.session_state:
        st.session_state[_tk] = format_typing_delay(CHARACTERS[_name].typing_delay)
    _think = f"tune_think_{_name}"
    if _think not in st.session_state:
        st.session_state[_think] = format_typing_delay(CHARACTERS[_name].think_delay)

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
        st.text_input(
            f"{_n} — think delay before typing (min,max sec)",
            key=f"tune_think_{_n}",
            help="Pause after the last message before this teammate starts typing.",
        )

with st.sidebar.expander("📝 Edit Scenario / Rules", expanded=False):
    scenario_content = st.text_area("Scenario Context", value=initial_scenario, height=150)
    behavioral_content = st.text_area("Behavioral Rules", value=initial_behavioral, height=150)

st.sidebar.divider()
st.sidebar.caption(
    "**STOP** pauses AI only (keeps chat; no API). **RESET** clears chat and counters."
)
col_run, col_stop, col_reset = st.sidebar.columns(3)
if col_run.button("▶ START"):
    st.session_state.sim_active = True
    st.session_state.next_ai_time = time.time() + START_WARMUP_SEC
    st.session_state.ai_burst_remaining = random.randint(2, 3)
    st.rerun()
if col_stop.button("⏸ STOP"):
    st.session_state.sim_active = False
    st.session_state.next_ai_time = 0.0
    st.session_state.ai_burst_remaining = 0
    st.rerun()
if col_reset.button("⏹ RESET"):
    st.session_state.sim_active = False
    st.session_state.messages = []
    st.session_state.next_ai_time = 0.0
    st.session_state.ai_burst_remaining = 0
    st.session_state.api_count = 0
    st.session_state.bot_turns_since_human = 0
    st.session_state.last_contribution_nudge_at_ai_count = -10_000
    st.rerun()

if st.sidebar.button("💾 EXPORT TRANSCRIPT"):
    if st.session_state.messages:
        df = pd.DataFrame(st.session_state.messages)
        df["participant_id"] = participant_id
        # Clear column order for reviewers; Excel needs UTF-8 BOM or it mangles curly quotes (â€™).
        _preferred = ["participant_id", "speaker", "text", "timestamp"]
        _cols = [c for c in _preferred if c in df.columns]
        _cols += [c for c in df.columns if c not in _cols]
        df = df[_cols]
        _buf = io.BytesIO()
        df.to_csv(_buf, index=False, encoding="utf-8-sig", lineterminator="\n")
        csv = _buf.getvalue()
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"study_{participant_id}.csv",
            mime="text/csv; charset=utf-8",
        )

# ==========================================
# 5. AI ENGINE (three distinct agents; one API call per turn)
# ==========================================
def run_agent_turn(speaker: str, extra_instruction: str | None = None) -> bool:
    """Append one AI line; return False on API failure (caller should not count the turn)."""
    agent = get_agent(speaker)
    try:
        txt, n_api = agent.generate_reply(
            client,
            MODEL_NAME,
            st.session_state.messages,
            leadership_style,
            scenario_content,
            behavioral_content,
            extra_instruction=extra_instruction,
        )
        st.session_state.api_count += n_api
        if not (txt or "").strip():
            return False
        st.session_state.messages.append({
            "speaker": speaker, "text": txt, "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        return True
    except Exception as e:
        st.error(f"AI Error ({speaker}): {e}")
        return False

# ==========================================
# 6. MAIN UI - INFO BAR
# ==========================================
_sim_status = "**Running**" if st.session_state.sim_active else "**Stopped** (AI off; transcript kept)"
st.info(
    f"Condition: **{leadership_style}** | Participant: **{participant_id}** | Sim: {_sim_status}"
)

st.markdown(
    _team_chat_css()
    + """
<style>
/* Paint Streamlit wrappers beige so empty chat is not a white slab */
div[data-testid="stMarkdownContainer"]:has(.tc-window) {
  background: #efeae2 !important;
  padding: 0 !important;
  margin: 0 !important;
  min-height: """
    + str(CHAT_SCROLL_HEIGHT_PX)
    + """px;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.tc-window) {
  background: #efeae2 !important;
}
@media (prefers-color-scheme: dark) {
  div[data-testid="stMarkdownContainer"]:has(.tc-window),
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.tc-window) {
    background: #0b141a !important;
  }
}
/* Composer: feels like Slack/Teams input bar */
div[data-testid="stChatInputContainer"] {
    position: sticky;
    bottom: 0;
    z-index: 999;
    padding-top: 0.5rem;
    margin-top: 0.75rem;
    background: var(--background-color, #ffffff);
    border-top: 1px solid rgba(0,0,0,0.08);
}
div[data-testid="stChatInputContainer"] textarea {
    border-radius: 12px !important;
    border: 1px solid #d1d5db !important;
    min-height: 2.75rem !important;
}
@media (prefers-color-scheme: dark) {
    div[data-testid="stChatInputContainer"] {
        background: var(--background-color, #0e1117);
        border-top-color: #333;
    }
    div[data-testid="stChatInputContainer"] textarea {
        border-color: #404040 !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 7. ASYNC CHAT UI (fragment timer; chat_input runs before AI work in the fragment)
# ==========================================
@st.fragment(run_every=FRAGMENT_REFRESH_SEC)
def chat_messages_panel():
    feed_h = max(120, CHAT_SCROLL_HEIGHT_PX - 72)
    with st.container(height=CHAT_SCROLL_HEIGHT_PX, border=False):
        st.markdown(
            _team_chat_message_html(
                st.session_state.messages,
                participant_id,
                feed_h,
            ),
            unsafe_allow_html=True,
        )
    if not st.session_state.sim_active and not st.session_state.messages:
        st.caption("Simulation not started. Use ▶ START in the sidebar.")
    elif not st.session_state.sim_active and st.session_state.messages:
        st.caption("Simulation **stopped** — transcript above. Click ▶ START to resume AI.")

    # Process participant input before any think/typing/API so the message shows on the next rerun
    # without waiting for the current bot turn to finish.
    if prompt := st.chat_input("Type a message"):
        st.session_state.messages.append({
            "speaker": "Participant",
            "text": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        st.session_state.bot_turns_since_human = 0
        if st.session_state.sim_active:
            st.session_state.next_ai_time = time.time() + random.uniform(*HUMAN_REPLY_DELAY_RANGE)
            # Let several bots respond in quick succession so the human isn’t dropped after one reply
            st.session_state.ai_burst_remaining = random.randint(3, 5)
        st.rerun()

    if not st.session_state.sim_active:
        return

    now = time.time()
    if now >= st.session_state.next_ai_time:
        if not st.session_state.messages:
            last_speaker, last_text = "Participant", ""
        else:
            last = st.session_state.messages[-1]
            last_speaker, last_text = last["speaker"], last.get("text", "")

        weight_map = {n: float(st.session_state[f"tune_w_{n}"]) for n in AI_NAMES}
        extra_instruction: str | None = None
        used_contribution_nudge = False
        current_ai_total = sum(
            1 for m in st.session_state.messages if m.get("speaker") in AI_NAMES
        )

        if last_speaker != "Participant" and _participant_invite_due(
            st.session_state.bot_turns_since_human,
            first=max(2, PARTICIPANT_INVITE_FIRST_BOT_TURNS),
            every=max(3, PARTICIPANT_INVITE_REPEAT_EVERY),
        ):
            speaker = "Zoe"
            cfg = CHARACTERS["Zoe"]
            extra_instruction = PARTICIPANT_INVITE_EXTRA
        else:
            nudge_plan = None
            _roll = float(QUIET_DRAW_IN_ROLL)
            if st.session_state.bot_turns_since_human < POST_HUMAN_BOT_WINDOW:
                _roll = 0.0
            if _roll > 0 and random.random() < _roll:
                nudge_plan = plan_contribution_nudge(
                    st.session_state.messages,
                    weight_map,
                    last_speaker=last_speaker,
                    last_text=last_text,
                    quiet_if_lines_below=QUIET_DRAW_IN_LINES_BELOW,
                    count_window=QUIET_DRAW_IN_WINDOW,
                    min_ai_messages=QUIET_DRAW_IN_MIN_AI,
                    address_lookback=QUIET_DRAW_IN_ADDRESS_LOOKBACK,
                    current_ai_total=current_ai_total,
                    ai_count_at_last_nudge=int(
                        st.session_state.last_contribution_nudge_at_ai_count
                    ),
                    cooldown_ai_messages=QUIET_DRAW_IN_COOLDOWN,
                )
            if nudge_plan is not None:
                speaker, _target, extra_instruction = nudge_plan
                cfg = CHARACTERS[speaker]
                used_contribution_nudge = True
            else:
                speaker, cfg = pick_next_speaker(
                    last_speaker,
                    last_text,
                    weights_override=weight_map,
                    recent_messages=st.session_state.messages,
                )

        think_raw = st.session_state.get(
            f"tune_think_{speaker}",
            format_typing_delay(cfg.think_delay),
        )
        t_lo, t_hi = parse_typing_delay(str(think_raw), cfg.think_delay)
        time.sleep(random.uniform(t_lo, t_hi))

        td_raw = st.session_state.get(
            f"tune_td_{speaker}",
            format_typing_delay(cfg.typing_delay),
        )
        lo, hi = parse_typing_delay(str(td_raw), cfg.typing_delay)

        with st.spinner(f"{speaker} is typing..."):
            time.sleep(random.uniform(lo, hi))
            ok = run_agent_turn(speaker, extra_instruction=extra_instruction)

        if not ok:
            st.session_state.next_ai_time = time.time() + 4.0
            return

        if used_contribution_nudge:
            st.session_state.last_contribution_nudge_at_ai_count = sum(
                1 for m in st.session_state.messages if m.get("speaker") in AI_NAMES
            )

        cfg = CHARACTERS[speaker]
        st.session_state.bot_turns_since_human += 1

        g0, g1 = cfg.burst_gap
        gap_after = random.uniform(g0, g1)

        if st.session_state.ai_burst_remaining > 0:
            st.session_state.next_ai_time = time.time() + gap_after
            st.session_state.ai_burst_remaining -= 1
        else:
            tail = st.session_state.messages[-6:]
            if any(m.get("speaker") == "Participant" for m in tail):
                idle_lo, idle_hi = PARTICIPANT_RECENT_IDLE_RANGE
            else:
                idle_lo, idle_hi = IDLE_AFTER_BURST_RANGE
            st.session_state.next_ai_time = time.time() + random.uniform(idle_lo, idle_hi)
            st.session_state.ai_burst_remaining = random.randint(1, 3)


chat_messages_panel()
