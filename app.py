import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import time

from maze_environment import MazeEnvironment
from q_learning import QLearningAgent
from mazes import get_maze


# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="GridExplorer | Q-Learning Maze Garden",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(213,235,194,.55), transparent 28%),
            radial-gradient(circle at 90% 25%, rgba(245,222,169,.35), transparent 25%),
            linear-gradient(135deg, #f7f6e9 0%, #edf5e5 52%, #f8f4e8 100%);
        color: #29452f;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e5f0db, #f5f0df);
        border-right: 1px solid #d2dfc8;
    }

    .hero {
        text-align: center;
        padding: 10px 0 22px;
    }

    .hero-title {
        font-size: 4rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -2px;
        color: #315d3a;
    }

    .hero-sub {
        margin-top: 10px;
        color: #6f806d;
        font-size: 1.12rem;
    }

    .card {
        background: rgba(255,255,255,.78);
        border: 1px solid #d7e3cb;
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 10px 28px rgba(67, 92, 57, .08);
        margin-bottom: 18px;
    }

    .eyebrow {
        color: #78906f;
        font-size: .82rem;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .section-title {
        color: #315c3a;
        font-size: 1.55rem;
        font-weight: 850;
        margin: 3px 0 8px;
    }

    .instruction {
        background: #eef6e8;
        border-left: 5px solid #75a36b;
        border-radius: 15px;
        padding: 14px 17px;
        margin: 12px 0 18px;
        color: #3f5d3e;
    }

    .step {
        display: inline-block;
        background: #f7fbf3;
        border: 1px solid #cfdfc5;
        border-radius: 999px;
        padding: 7px 12px;
        margin: 3px 4px 3px 0;
        font-size: .88rem;
        color: #4e684c;
    }

    .step-active {
        background: #dcefd2;
        border-color: #8db57f;
        font-weight: 800;
    }

    .preset {
        background: rgba(255,255,255,.82);
        border: 1px solid #d7e3cb;
        border-radius: 20px;
        padding: 18px;
        min-height: 150px;
        box-shadow: 0 7px 20px rgba(67, 92, 57, .06);
    }

    .preset h3 {
        margin: 0 0 6px;
        color: #315d3a;
    }

    .preset p {
        color: #71806e;
        font-size: .92rem;
        margin-bottom: 10px;
    }

    .legend {
        display: flex;
        gap: 14px;
        justify-content: center;
        flex-wrap: wrap;
        color: #657461;
        font-size: .9rem;
        margin: 10px 0 4px;
    }

    .legend span {
        background: rgba(255,255,255,.65);
        border: 1px solid #d8e3ce;
        border-radius: 999px;
        padding: 5px 10px;
    }

    .metric-box {
        background: rgba(255,255,255,.72);
        border: 1px solid #d7e3cb;
        border-radius: 18px;
        padding: 15px;
        text-align: center;
    }

    .metric-number {
        color: #376441;
        font-size: 1.65rem;
        font-weight: 900;
    }

    .metric-label {
        color: #7a8775;
        font-size: .82rem;
    }

    .success {
        background: #e6f4df;
        border-left: 5px solid #62a45c;
        border-radius: 15px;
        padding: 15px 18px;
        color: #315d3a;
        font-weight: 700;
        margin: 12px 0;
    }

    .warning {
        background: #fff4da;
        border-left: 5px solid #d3a04a;
        border-radius: 15px;
        padding: 15px 18px;
        color: #765d2d;
        margin: 12px 0;
    }

    .footer {
        text-align: center;
        color: #899484;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #dce5d5;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid #b9cda9;
        background: #f9fcf5;
        color: #365e3d;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #79a36e;
        background: #eaf5e3;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.68);
        border: 1px solid #d7e3cb;
        padding: 10px;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
def blank_grid():
    return np.array([["."] * 7 for _ in range(7)], dtype=str)


defaults = {
    "mode": "Preset Mazes",
    "selected_maze": "Easy Garden",
    "custom_grid": blank_grid(),
    "custom_click_count": 0,
    "custom_start": None,
    "custom_goal": None,
    "custom_obstacles": [],
    "maze": None,
    "agent": None,
    "training_data": None,
    "learned_path": [],
    "training_complete": False,
    "last_training_config": None,
    "animate_training": True,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_training():
    st.session_state.agent = None
    st.session_state.training_data = None
    st.session_state.learned_path = []
    st.session_state.training_complete = False
    st.session_state.last_training_config = None


def reset_custom():
    st.session_state.custom_grid = blank_grid()
    st.session_state.custom_click_count = 0
    st.session_state.custom_start = None
    st.session_state.custom_goal = None
    st.session_state.custom_obstacles = []
    st.session_state.maze = None
    reset_training()


def render_maze(maze, path=None, agent_position=None, size=58):
    """Render the maze in a fixed HTML iframe so the grid never collapses."""
    if maze is None:
        return

    path_set = set(path or [])
    rows, cols = maze.shape
    gap = 6
    padding = 14
    cell = size
    frame_w = cols * cell + (cols - 1) * gap + padding * 2 + 10
    frame_h = rows * cell + (rows - 1) * gap + padding * 2 + 10

    cells = []
    for r in range(rows):
        for c in range(cols):
            value = maze[r, c]
            bg = "linear-gradient(145deg,#dcebc9,#cfe3ba)"
            border = "#b9cf9f"
            content = "🌿"
            label = "Grass"

            if value == "#":
                bg = "linear-gradient(145deg,#7c7058,#625941)"
                border = "#514832"
                content = "🪨"
                label = "Obstacle"
            elif value == "S":
                bg = "linear-gradient(145deg,#bfe39f,#91c878)"
                border = "#4f8544"
                content = "🟢"
                label = "Start"
            elif value == "G":
                bg = "linear-gradient(145deg,#ffe1a6,#f2c86f)"
                border = "#c29435"
                content = "🌸"
                label = "Goal"
            elif (r, c) in path_set:
                bg = "linear-gradient(145deg,#d3edbd,#a9d68d)"
                border = "#74a75d"
                content = "🌱"
                label = "Learned path"

            if agent_position == (r, c):
                bg = "linear-gradient(145deg,#cbe7f6,#94c9e4)"
                border = "#4e88a9"
                content = "🐝"
                label = "AI agent"

            cells.append(
                f'<div title="{label} | Row {r+1}, Column {c+1}" '
                f'style="width:{cell}px;height:{cell}px;box-sizing:border-box;'
                f'display:flex;align-items:center;justify-content:center;'
                f'border-radius:14px;background:{bg};border:2px solid {border};'
                f'font-size:{int(cell*0.42)}px;line-height:1;">{content}</div>'
            )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
* {{ box-sizing:border-box; }}
body {{ margin:0; background:transparent; font-family:Arial,sans-serif; }}
.frame {{ width:{frame_w}px; height:{frame_h}px; margin:8px auto; padding:{padding}px;
          background:linear-gradient(145deg,#86a96d,#6f905a); border:5px solid #5d7d4a;
          border-radius:24px; box-shadow:0 14px 30px rgba(53,76,45,.18), inset 0 0 0 2px rgba(255,255,255,.18); }}
.grid {{ display:grid; grid-template-columns:repeat({cols},{cell}px); gap:{gap}px; }}
</style></head><body>
<div class="frame"><div class="grid">{''.join(cells)}</div></div>
</body></html>"""

    components.html(html, height=frame_h + 25, scrolling=False)


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">🌿Grid Explorer</div>
      <div class="hero-sub">
        A tiny garden where an Agent learns its way home 🐝
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🌱 Maze Garden")
    st.caption("Choose a world, tune the learner, and watch it discover a path.")

    mode = st.radio(
        "Choose a mode",
        ["Preset Mazes", "Build Your Own"],
        index=0 if st.session_state.mode == "Preset Mazes" else 1,
    )
    st.session_state.mode = mode

    st.markdown("---")
    st.markdown("### 🧠 Q-Learning Settings")

    learning_rate = st.slider("Learning Rate (α)", 0.01, 1.0, 0.10, 0.01)
    discount_factor = st.slider("Discount Factor (γ)", 0.50, 0.99, 0.95, 0.01)
    episodes = st.slider("Training Episodes", 100, 3000, 1000, 100)
    max_steps = st.slider("Maximum Steps / Episode", 50, 500, 200, 10)
    epsilon_decay = st.slider("Exploration Decay", 0.900, 0.999, 0.995, 0.001)

    st.session_state.animate_training = st.checkbox(
        "🎬 Show learning replay",
        value=st.session_state.animate_training,
    )

    st.markdown("---")
    st.caption("💡 α controls learning speed. γ controls how much future rewards matter. ε controls exploration.")


# ============================================================
# PRESET MAZES
# ============================================================
if mode == "Preset Mazes":
    st.markdown(
        '<div class="card"><div class="eyebrow">Choose your world</div>'
        '<div class="section-title">🌳 Garden Mazes</div>'
        '<div style="color:#71806e;">Start with a handcrafted maze, then let the Q-learning agent discover the route.</div></div>',
        unsafe_allow_html=True,
    )

    preset_info = {
        # EASY
        "Easy Garden": (
            "🌱",
            "A gentle first garden",
            "Few walls • great for understanding the basics"
        ),
        "Easy Meadow": (
            "🌼",
            "A peaceful meadow",
            "Simple turns • a friendly introduction"
        ),
        "Easy Breeze": (
            "🍃",
            "The breezy garden",
            "Light obstacles • easy exploration"
        ),
        "Easy Flower Path": (
            "🌸",
            "The flower trail",
            "A few barriers • follow the open garden"
        ),
        "Easy Little Grove": (
            "🌿",
            "A tiny grove",
            "Small detours • gentle Q-learning challenge"
        ),

        # MEDIUM
        "Medium Garden": (
            "🌿",
            "A winding little world",
            "More barriers • stronger exploration required"
        ),
        "Medium Winding Garden": (
            "🌳",
            "The winding garden",
            "Several turns • the agent must explore carefully"
        ),
        "Medium Twisted Path": (
            "🌀",
            "The twisted trail",
            "Tighter routes • more exploration required"
        ),
        "Medium Hedge Maze": (
            "🌱",
            "The hedge maze",
            "Dense barriers • careful decisions matter"
        ),
        "Medium Forest Trail": (
            "🌲",
            "The forest trail",
            "Multiple turns • a stronger learning challenge"
        ),

        # HARD
        "Hard Garden": (
            "🌲",
            "The tangled grove",
            "Tricky turns • a tougher learning challenge"
        ),
        "Hard Tangled Grove": (
            "🌳",
            "The tangled grove",
            "Dense barriers • exploration becomes important"
        ),
        "Hard Deep Forest": (
            "🌲",
            "The deep forest",
            "Narrow routes • difficult exploration"
        ),
        "Hard Chaos Garden": (
            "🍂",
            "The chaotic garden",
            "Complex barriers • a serious Q-learning challenge"
        ),
        "Hard Mountain Trail": (
            "⛰️",
            "The mountain trail",
            "Tight passages • the hardest garden challenge"
        ),
    }

    # --------------------------------------------------------
    # Difficulty selector
    # --------------------------------------------------------

    difficulty = st.selectbox(
        "🌿 Choose difficulty",
        ["Easy", "Medium", "Hard"],
        index=0,
    )

    difficulty_mazes = [
        name
        for name in preset_info
        if name.startswith(difficulty)
    ]

    selected_maze = st.selectbox(
        "🌱 Choose a maze",
        difficulty_mazes,
        index=0,
    )

    icon, title, desc = preset_info[selected_maze]

    st.markdown(
        f"""
        <div class="card">
            <div class="eyebrow">Selected world</div>
            <div class="section-title">
                {icon} {selected_maze}
            </div>
            <div style="color:#71806e;">
                <b>{title}</b><br>
                {desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load selected maze whenever the selection changes
    if st.session_state.selected_maze != selected_maze:
        st.session_state.selected_maze = selected_maze
        st.session_state.maze = get_maze(selected_maze)
        reset_training()

    if st.session_state.maze is None:
        st.session_state.maze = get_maze(selected_maze)

    render_maze(st.session_state.maze)

    st.markdown(
        """
        <div class="legend">
          <span>🟢 Start</span>
          <span>🌸 Goal</span>
          <span>🪨 Wall</span>
          <span>🌱 Learned path</span>
          <span>🐝 AI agent</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CUSTOM MAZE BUILDER
# ============================================================
else:
    st.markdown(
        '<div class="card"><div class="eyebrow">Create a world</div>'
        '<div class="section-title">🛠️ Build Your Own Garden Maze</div>'
        '<div style="color:#71806e;">Design a 7×7 world and give the AI exactly six obstacles to navigate around.</div></div>',
        unsafe_allow_html=True,
    )

    count = st.session_state.custom_click_count

    if count == 0:
        active = "step-active"
    else:
        active = ""

    st.markdown(
        f"""
        <div class="instruction">
          <b>Build order</b><br>
          <span class="step {active}">🟢 Click Start</span>
          <span class="step">🌸 Click Goal</span>
          <span class="step">🪨 Click exactly 6 Obstacles</span>
          <br><small>Every click has a purpose. Start and Goal cannot overlap, and obstacle cells cannot repeat.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if count == 0:
        st.info("🌱 Step 1 of 8 — choose the Start cell.")
    elif count == 1:
        st.info("🌸 Step 2 of 8 — choose the Goal cell.")
    elif count < 8:
        st.info(f"🪨 Step {count + 1} of 8 — choose obstacle {count - 1} of 6.")
    else:
        st.success("✨ All 8 selections are complete. Your garden is ready to build!")

    # Builder grid
    for r in range(7):
        cols_ui = st.columns(7, gap="small")
        for c in range(7):
            value = st.session_state.custom_grid[r, c]
            label = {"S": "🟢", "G": "🌸", "#": "🪨", ".": "🌿"}[value]

            with cols_ui[c]:
                if st.button(
                    label,
                    key=f"builder_{r}_{c}",
                    use_container_width=True,
                ):
                    pos = (r, c)
                    count_now = st.session_state.custom_click_count

                    if count_now == 0:
                        st.session_state.custom_start = pos
                        st.session_state.custom_grid[r, c] = "S"
                        st.session_state.custom_click_count = 1
                        reset_training()
                        st.rerun()

                    elif count_now == 1:
                        if pos == st.session_state.custom_start:
                            st.warning("🌱 Start and Goal must be different cells.")
                        else:
                            st.session_state.custom_goal = pos
                            st.session_state.custom_grid[r, c] = "G"
                            st.session_state.custom_click_count = 2
                            reset_training()
                            st.rerun()

                    elif 2 <= count_now < 8:
                        occupied = [
                            st.session_state.custom_start,
                            st.session_state.custom_goal,
                        ] + st.session_state.custom_obstacles

                        if pos in occupied:
                            st.warning("🌿 That cell is already occupied. Choose another cell.")
                        else:
                            st.session_state.custom_obstacles.append(pos)
                            st.session_state.custom_grid[r, c] = "#"
                            st.session_state.custom_click_count += 1
                            reset_training()
                            st.rerun()

    b1, b2 = st.columns([2, 1])

    with b1:
        if st.button(
            "✨ Build This Maze",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.custom_click_count != 8),
        ):
            candidate = st.session_state.custom_grid.copy()
            env = MazeEnvironment(candidate)

            if env.is_solvable():
                st.session_state.maze = candidate
                reset_training()
                st.success("🌸 Your maze is solvable! It is ready for the AI.")
            else:
                st.session_state.maze = None
                st.error("🍂 This maze has no path from Start to Goal. Move an obstacle and try again.")

    with b2:
        if st.button("🔄 Clear Maze", use_container_width=True):
            reset_custom()
            st.rerun()

    if st.session_state.maze is not None:
        st.markdown(
            '<div class="card"><div class="eyebrow">Maze preview</div>'
            '<div class="section-title">🌿 Your Garden</div></div>',
            unsafe_allow_html=True,
        )
        render_maze(st.session_state.maze)
        st.markdown(
            '<div class="legend"><span>🟢 Start</span><span>🌸 Goal</span>'
            '<span>🪨 Wall</span><span>🌱 Learned path</span><span>🐝 AI agent</span></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# LEARNING REPLAY
# ============================================================
def render_learning_replay(maze, episode_trajectories):
    """Show a browser-based replay of representative training episodes.

    The maze styling, colors, emojis and grid dimensions intentionally match
    the existing WhimsyGrid maze renderer. Only the agent position changes.
    """
    if not episode_trajectories:
        return

    rows, cols = maze.shape
    cell = 58
    gap = 6
    padding = 14
    frame_w = cols * cell + (cols - 1) * gap + padding * 2 + 10
    frame_h = rows * cell + (rows - 1) * gap + padding * 2 + 10

    ordered = sorted(episode_trajectories.items(), key=lambda item: item[0])
    frames = []

    # Reuse the exact visual language of render_maze.
    base_cells = []
    for r in range(rows):
        for c in range(cols):
            value = maze[r, c]
            bg = "linear-gradient(145deg,#dcebc9,#cfe3ba)"
            border = "#b9cf9f"
            content = "🌿"
            label = "Grass"

            if value == "#":
                bg = "linear-gradient(145deg,#7c7058,#625941)"
                border = "#514832"
                content = "🪨"
                label = "Obstacle"
            elif value == "S":
                bg = "linear-gradient(145deg,#bfe39f,#91c878)"
                border = "#4f8544"
                content = "🟢"
                label = "Start"
            elif value == "G":
                bg = "linear-gradient(145deg,#ffe1a6,#f2c86f)"
                border = "#c29435"
                content = "🌸"
                label = "Goal"

            base_cells.append((r, c, bg, border, content, label))

    for episode, trajectory in ordered:
        # Keep the replay smooth without making very long exploratory episodes huge.
        sampled = trajectory[::2] if len(trajectory) > 90 else trajectory
        if not sampled or sampled[-1] != trajectory[-1]:
            sampled.append(trajectory[-1])

        for position in sampled:
            cells = []
            for r, c, bg, border, content, label in base_cells:
                if position == (r, c):
                    bg = "linear-gradient(145deg,#cbe7f6,#94c9e4)"
                    border = "#4e88a9"
                    content = "🐝"
                    label = "AI agent"

                cells.append(
                    f'<div title="{label} | Row {r+1}, Column {c+1}" '
                    f'style="width:{cell}px;height:{cell}px;box-sizing:border-box;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'border-radius:14px;background:{bg};border:2px solid {border};'
                    f'font-size:{int(cell*0.42)}px;line-height:1;">{content}</div>'
                )

            frames.append({
                "episode": int(episode),
                "html": "".join(cells),
            })

    if not frames:
        return

    import json
    frames_json = json.dumps(frames, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
* {{ box-sizing:border-box; }}
body {{ margin:0; background:transparent; font-family:Arial,sans-serif; }}
.title {{ text-align:center; color:#315d3a; font-weight:800; font-size:18px; margin:4px 0 8px; }}
.info {{ text-align:center; color:#6f806d; font-size:13px; margin-bottom:8px; }}
.frame {{ width:{frame_w}px; height:{frame_h}px; margin:0 auto; padding:{padding}px;
          background:linear-gradient(145deg,#86a96d,#6f905a); border:5px solid #5d7d4a;
          border-radius:24px; box-shadow:0 14px 30px rgba(53,76,45,.18), inset 0 0 0 2px rgba(255,255,255,.18); }}
.grid {{ display:grid; grid-template-columns:repeat({cols},{cell}px); gap:{gap}px; }}
</style></head><body>
<div class="title">🎬 AI Learning Replay</div>
<div class="info" id="info">Starting exploration...</div>
<div class="frame"><div class="grid" id="grid"></div></div>
<script>
const frames = {frames_json};
const grid = document.getElementById('grid');
const info = document.getElementById('info');
let i = 0;
function showFrame() {{
    const f = frames[i];
    grid.innerHTML = f.html;
    info.textContent = `Episode ${{f.episode}} • The agent is exploring and learning`;
    i = (i + 1) % frames.length;
}}
showFrame();
setInterval(showFrame, 130);
</script></body></html>"""

    components.html(html, height=frame_h + 80, scrolling=False)


# ============================================================
# TRAINING AREA
# ============================================================
maze = st.session_state.maze

if maze is not None:
    env = MazeEnvironment(maze)

    if not env.is_solvable():
        st.markdown(
            '<div class="warning">🍂 This maze is currently unsolvable. Please choose a different maze or rebuild it.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown(
        '<div class="card"><div class="eyebrow">Learning laboratory</div>'
        '<div class="section-title">🧠 Train the Garden Agent</div>'
        '<div style="color:#71806e;">'
        'The agent starts with an empty Q-table and gradually learns which actions lead it closer to the flower.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    shortest = env.shortest_path_bfs()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-number">{env.rows}×{env.cols}</div>'
            '<div class="metric-label">Maze size</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-number">{len(shortest)-1}</div>'
            '<div class="metric-label">Shortest path</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-box"><div class="metric-number">{env.num_states}</div>'
            '<div class="metric-label">States</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-box"><div class="metric-number">{env.num_actions}</div>'
            '<div class="metric-label">Actions</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    if st.session_state.training_complete:
        st.markdown(
            '<div class="success">🌸 The AI has already trained on this maze. '
            'You can retrain with different hyperparameters or watch the learned solution below.</div>',
            unsafe_allow_html=True,
        )

    train_clicked = st.button(
        "🧠 Train Q-Learning Agent",
        type="primary",
        use_container_width=True,
    )

    if train_clicked:
        reset_training()

        agent = QLearningAgent(
            env,
            alpha=learning_rate,
            gamma=discount_factor,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=epsilon_decay,
        )

        progress = st.progress(0)
        status = st.empty()

        def training_callback(episode, reward, steps, epsilon, reached_goal):
            # Keep UI responsive; do not redraw for every episode.
            if episode == 1 or episode % max(1, episodes // 100) == 0 or episode == episodes:
                progress.progress(min(episode / episodes, 1.0))
                status.info(
                    f"🌱 Learning episode {episode}/{episodes}  •  "
                    f"Reward: {reward:.1f}  •  Steps: {steps}  •  "
                    f"ε: {epsilon:.3f}"
                )

        result = agent.train(
            episodes=episodes,
            max_steps=max_steps,
            callback=training_callback,
        )

        progress.progress(1.0)
        status.success("🌸 Training complete! The Q-table has been learned.")

        st.session_state.agent = agent
        st.session_state.training_data = result
        st.session_state.learned_path = agent.get_learned_path()
        st.session_state.training_complete = True
        st.session_state.last_training_config = {
            "alpha": learning_rate,
            "gamma": discount_factor,
            "episodes": episodes,
            "max_steps": max_steps,
            "epsilon_decay": epsilon_decay,
        }

        st.rerun()


# ============================================================
# RESULTS
# ============================================================
if st.session_state.training_complete and st.session_state.agent is not None:
    agent = st.session_state.agent
    data = st.session_state.training_data
    learned_path = st.session_state.learned_path
    shortest = MazeEnvironment(maze).shortest_path_bfs()

    reached_goal = len(learned_path) > 0 and learned_path[-1] == tuple(
        MazeEnvironment(maze).find_symbol("G")
    )

    st.markdown(
        '<div class="card"><div class="eyebrow">What the AI discovered</div>'
        '<div class="section-title">🌸 Learned Solution</div></div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Learned steps", max(0, len(learned_path) - 1))
    with r2:
        st.metric("Shortest possible", max(0, len(shortest) - 1))
    with r3:
        st.metric("Final reward", f"{data['rewards'][-1]:.1f}")
    with r4:
        st.metric("Goal reached", "YES 🌸" if reached_goal else "NO 🍂")

    if reached_goal:
        render_maze(maze, path=learned_path)

        if st.session_state.animate_training:
            st.markdown("### 🎬 Watch the AI learn")
            render_learning_replay(maze, data.get("episode_trajectories", {}))
            st.caption("The replay shows representative episodes from early exploration to later learning.")

        actions = agent.get_path_actions()
        action_text = " → ".join(actions) if actions else "No actions"

        st.markdown(
            f'<div class="success"><b>Action sequence:</b> {action_text}</div>',
            unsafe_allow_html=True,
        )

        path_len = len(learned_path) - 1
        shortest_len = len(shortest) - 1

        if path_len == shortest_len:
            st.success("🏆 The learned route matches the true shortest path!")
        else:
            st.info(
                f"🌱 The agent found a valid route of {path_len} steps. "
                f"The shortest route is {shortest_len} steps."
            )
    else:
        st.warning(
            "The learned policy did not reach the goal yet. "
            "Try more episodes, a slower exploration decay, or a different learning rate."
        )

    # --------------------------------------------------------
    # Learning curves
    # --------------------------------------------------------
    st.markdown("### 📈 Learning Curves")

    chart_df = pd.DataFrame(
        {
            "Episode": np.arange(1, len(data["rewards"]) + 1),
            "Reward": data["rewards"],
            "Steps": data["steps"],
        }
    ).set_index("Episode")

    st.line_chart(chart_df[["Reward"]], use_container_width=True)
    st.caption("Reward per episode — successful learning should generally produce stronger returns.")

    st.line_chart(chart_df[["Steps"]], use_container_width=True)
    st.caption("Steps per episode — as the policy improves, successful routes tend to become shorter.")

    # --------------------------------------------------------
    # Q table
    # --------------------------------------------------------
    with st.expander("🔬 Inspect the learned Q-table"):
        q_df = pd.DataFrame(
            agent.q_table,
            columns=["↑ Up", "↓ Down", "← Left", "→ Right"],
        )
        q_df.index.name = "State"
        st.dataframe(q_df.round(3), use_container_width=True)

    with st.expander("🗺️ Inspect the learned policy"):
        policy = agent.get_policy()
        st.code(policy)

    with st.expander("📚 How Q-learning works"):
        st.markdown(
            """
            **Q-learning** learns the expected future value of taking an action from each state.

            The update rule used by WhimsyGrid is:

            **Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]**

            - **Q(s,a):** current value of taking action `a` in state `s`
            - **α:** learning rate
            - **r:** reward received after the action
            - **γ:** discount factor
            - **s':** next state
            - **ε:** exploration probability used to balance trying new actions and using learned actions

            The agent begins with no knowledge of the maze. Across many episodes,
            rewards update the Q-table until useful actions receive higher values.
            """
        )


# ============================================================
# EMPTY STATE
# ============================================================
elif maze is None:
    st.markdown(
        """
        <div class="card" style="text-align:center;padding:55px 25px;">
          <div style="font-size:4rem;">🌿 🐝 🌸</div>
          <div class="section-title">Your garden is waiting</div>
          <div style="color:#71806e;">
            Choose a preset maze above or build your own garden.
            Once a maze is ready, the <b>Train Q-Learning Agent</b> button will appear here.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
      🌿 Grid Explorer • Tabular Q-Learning • Classic Control & Environments<br>
      <small>From empty Q-table → exploration → learning → shortest-path discovery</small>
      <small>Created by Vempali Hrishita</small>

    </div>
    """,
    unsafe_allow_html=True,
)
