import os
import streamlit as st
from dotenv import load_dotenv
from datetime import date, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()

WORKER_MODEL = "gemini-2.5-flash-lite"
AUDIT_MODEL  = "gemini-3-pro-preview"

FIBONACCI  = [1, 2, 3, 5, 8, 13]
PRIORITIES = ["Critical", "High", "Medium", "Low"]

EXAMPLE_PROMPTS = [
    {"label": "🛒 E-commerce checkout",  "text": "Implement a multi-step checkout flow with cart summary, address form, payment via Stripe, and order confirmation email. Must support guest and authenticated users."},
    {"label": "🔐 Auth system",           "text": "Build user authentication: registration, login, JWT tokens, refresh token rotation, password reset via email, and Google OAuth2 integration."},
    {"label": "📊 Analytics dashboard",   "text": "Create a real-time analytics dashboard showing DAU, revenue, conversion funnel, and cohort retention. Data from PostgreSQL, charts via Chart.js, exportable to CSV."},
    {"label": "🔔 Notification service",  "text": "Design a notification microservice supporting push, email, and SMS channels. Users can manage preferences. Events triggered via Kafka. Delivery tracking and retry logic required."},
]

gemini_api_key = os.getenv("GEMINI_API_KEY")
llm_worker = ChatGoogleGenerativeAI(model=WORKER_MODEL, temperature=0.1, google_api_key=gemini_api_key)
llm_audit  = ChatGoogleGenerativeAI(model=AUDIT_MODEL,  temperature=0.1, google_api_key=gemini_api_key)

JIRA_SERVER  = os.getenv("JIRA_SERVER", "")
JIRA_EMAIL   = os.getenv("JIRA_EMAIL", "")
JIRA_TOKEN   = os.getenv("JIRA_TOKEN", "")
PROJECT_KEY  = os.getenv("JIRA_PROJECT_KEY", "")
SP_FIELD     = os.getenv("JIRA_SP_FIELD", "customfield_10016")
JIRA_BOARD_ID = os.getenv("JIRA_BOARD_ID", "")

def connect_jira(server, email, token):
    from jira import JIRA
    return JIRA(server=server, basic_auth=(email, token))

def get_boards(jira, project_key):
    """Get all Scrum boards for a project."""
    try:
        boards = jira.boards(projectKeyOrID=project_key, type="scrum")
        return [(b.id, b.name) for b in boards]
    except:
        return []

def create_sprint(jira, board_id, sprint_name, start_dt, end_dt):
    """Create a sprint on the board and return its ID."""
    sprint = jira.create_sprint(
        name=sprint_name,
        board_id=board_id,
        startDate=start_dt.strftime("%Y-%m-%dT09:00:00.000+0000"),
        endDate=end_dt.strftime("%Y-%m-%dT18:00:00.000+0000"),
    )
    return sprint.id

def move_to_sprint(jira, sprint_id, issue_keys):
    """Move issues into a sprint."""
    if issue_keys:
        jira.add_issues_to_sprint(sprint_id, issue_keys)

# ==========================================
# SCHEMAS
# ==========================================
class UserStory(BaseModel):
    summary: str = Field(description="User story: 'As a [user], I want [action] so that [value]'")
    ac: str      = Field(description="Acceptance criteria")

class UserStoryList(BaseModel):
    stories: List[UserStory]

class Risk(BaseModel):
    title:       str
    description: str
    severity:    Literal["Critical", "High", "Medium", "Low"]
    mitigation:  str

class RiskAnalysis(BaseModel):
    risks:              List[Risk]
    overall_risk_level: Literal["Critical", "High", "Medium", "Low"]
    summary:            str

class TechTask(BaseModel):
    summary:             str
    description:         str
    acceptance_criteria: str
    sp:                  int = Field(default=3)
    subtasks:            List[str]
    priority:            Literal["Critical", "High", "Medium", "Low"] = Field(default="Medium")
    priority_reason:     str = Field(default="")

class TechTaskList(BaseModel):
    tasks: List[TechTask]

class AuditResult(BaseModel):
    status:  Literal["PASS", "FAIL"]
    culprit: Literal["PM", "ARCHITECT", "NONE"]
    report:  str

class SprintTask(BaseModel):
    task_summary:  str       = Field(description="Exact task summary from input")
    sprint_number: int       = Field(description="Sprint number (1-based)")
    dependencies:  List[str] = Field(default=[])
    reason:        str       = Field(description="Why this sprint")

class SprintPlan(BaseModel):
    sprint_tasks:   List[SprintTask]
    sprint_goals:   List[str] = Field(description="One goal per sprint")
    planning_notes: str

# ==========================================
# CHAINS
# ==========================================
pm_chain = ChatPromptTemplate.from_messages([
    ("system", "Act as a PM. If Global AC is 'None', generate 3-5 AC per story. {feedback_context}"),
    ("human",  "Requirements: {requirements}\nGlobal AC: {global_ac}")
]) | llm_worker.with_structured_output(UserStoryList)

risk_chain = ChatPromptTemplate.from_messages([
    ("system", "Act as a Risk Analyst. Find technical, security, performance, integration, and business risks. Be specific and actionable."),
    ("human",  "Requirements: {requirements}\nStories: {stories}")
]) | llm_audit.with_structured_output(RiskAnalysis)

arch_chain = ChatPromptTemplate.from_messages([
    ("system", """Act as Lead Architect. Decompose stories into tasks.
MANDATORY:
1. Copy PM's 'ac' exactly into 'acceptance_criteria'.
2. Assign priority: Critical=blocks others, High=core feature, Medium=standard, Low=nice-to-have.
3. Write priority_reason for each task.
Risk context: {risk_context}
{feedback_context}"""),
    ("human", "Stories: {stories}")
]) | llm_worker.with_structured_output(TechTaskList)

audit_chain = ChatPromptTemplate.from_messages([
    ("system", "Act as QA Auditor. FAIL if AC, technical depth, or priority reasoning is missing."),
    ("human",  "Req: {req}\nPM Stories: {pm_output}\nTech Plan: {tech_plan}")
]) | llm_audit.with_structured_output(AuditResult)

sprint_chain = ChatPromptTemplate.from_messages([
    ("system", """Act as Agile Sprint Planner. Distribute tasks across sprints.
Rules:
- Sprint capacity = {velocity} story points
- Sprint duration = {sprint_days} days
- Start date = {start_date}
- Respect dependencies (dependent tasks go in later sprints)
- Critical/High priority tasks first
- Balance load evenly
- Each sprint has a clear deliverable goal"""),
    ("human", "Tasks:\n{tasks}")
]) | llm_audit.with_structured_output(SprintPlan)

# ==========================================
# SESSION STATE
# ==========================================
st.set_page_config(page_title="AI Sprint Architect", layout="wide")

for key, default in [
    ("workflow_result", None), ("risk_result", None),
    ("sprint_plan", None),     ("story_id_value", SP_FIELD),
    ("req_text", ""),          ("board_id", JIRA_BOARD_ID),
    ("boards_list", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

PRIORITY_COLOR = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#22c55e"}
SEVERITY_ICON  = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
SPRINT_COLORS  = ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]
JIRA_PRI_MAP   = {"Critical": "Highest", "High": "High", "Medium": "Medium", "Low": "Low"}

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Mode")
    jira_mode = st.toggle("Enable Jira Integration", value=bool(JIRA_SERVER))

    if jira_mode:
        st.divider()
        st.subheader("🔑 Jira Connection")
        env_ok = all([JIRA_SERVER, JIRA_EMAIL, JIRA_TOKEN, PROJECT_KEY])
        if env_ok:
    st.success("✅ Loaded from .env")
else:
    st.warning("⚠️ .env incomplete")
        jira_server  = st.text_input("Server URL",  value=JIRA_SERVER)
        jira_email   = st.text_input("Email",        value=JIRA_EMAIL)
        jira_token   = st.text_input("API Token",    value=JIRA_TOKEN, type="password")
        project_key  = st.text_input("Project Key",  value=PROJECT_KEY)

        with st.expander("🛠️ Advanced"):
            if st.button("Detect Story Points Field", use_container_width=True):
                try:
                    tmp  = connect_jira(jira_server, jira_email, jira_token)
                    sp_f = [f for f in tmp.fields() if 'story point' in f['name'].lower()]
                    if sp_f:
                        st.session_state.story_id_value = sp_f[0]['id']
                        st.success(f"Detected: {sp_f[0]['id']}")
                except: st.error("Detection failed.")
            st.text_input("SP Field ID", value=st.session_state.story_id_value, disabled=True)

        st.divider()
        st.subheader("🏃 Sprint Board")

        # Board detection
        col_board, col_detect = st.columns([3, 2])
        with col_board:
            board_input = st.text_input(
                "Board ID",
                value=st.session_state.board_id,
                placeholder=JIRA_BOARD_ID or "e.g. 42"
            )
            st.session_state.board_id = board_input
        with col_detect:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🔍 Detect", use_container_width=True):
                try:
                    tmp    = connect_jira(jira_server, jira_email, jira_token)
                    boards = get_boards(tmp, project_key)
                    if boards:
                        st.session_state.boards_list = boards
                        # Auto-select first board
                        st.session_state.board_id = str(boards[0][0])
                        st.success(f"Found {len(boards)} board(s)")
                    else:
                        st.warning("No Scrum boards found.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Show detected boards as selector if multiple found
        if len(st.session_state.boards_list) > 1:
            board_options = {f"{b[1]} (ID: {b[0]})": str(b[0]) for b in st.session_state.boards_list}
            selected_board_label = st.selectbox("Select Board", list(board_options.keys()))
            st.session_state.board_id = board_options[selected_board_label]

        if st.session_state.board_id:
            st.success(f"✅ Board ID: {st.session_state.board_id}")

        create_sprints = st.checkbox("Auto-create sprints in Jira", value=True,
                                     help="Creates sprints with dates from the AI timeline and assigns tasks automatically")
        assign_to_epic = st.checkbox("Assign to Epic", value=True)

    else:
        jira_server = jira_email = jira_token = project_key = ""
        assign_to_epic  = False
        create_sprints  = False
        st.info("🧪 Test mode — no Jira needed.")

    st.divider()
    st.subheader("🤖 Pipeline Config")
    use_global_ac     = st.checkbox("Add Custom Acceptance Criteria", value=False)
    global_ac_text    = st.text_area("Criteria details", height=80) if use_global_ac else ""
    create_user_story = st.checkbox("Create User Story (PM stage)", value=True)

    st.divider()
    st.subheader("📅 Sprint Planner")
    velocity    = st.slider("Velocity (SP/sprint)", 8, 60, 20, step=2)
    sprint_days = st.selectbox("Sprint duration", [7, 14, 21], index=1,
                               format_func=lambda x: f"{x} days ({x//7} week{'s' if x>7 else ''})")
    start_date  = st.date_input("Project start date", value=date.today())

    st.divider()
    if st.button("🗑️ Reset Session", use_container_width=True):
        for k in ["workflow_result", "risk_result", "sprint_plan", "boards_list"]:
            st.session_state[k] = None
        st.session_state.req_text = ""
        st.rerun()

    st.divider()
    st.caption(f"Worker: `{WORKER_MODEL}`")
    st.caption(f"Auditor: `{AUDIT_MODEL}`")

# ==========================================
# MAIN
# ==========================================
st.title("🚀 AI Sprint Architect")

st.markdown("**💡 Example prompts:**")
ex_cols = st.columns(len(EXAMPLE_PROMPTS))
for i, ex in enumerate(EXAMPLE_PROMPTS):
    with ex_cols[i]:
        if st.button(ex["label"], use_container_width=True, key=f"ex_{i}"):
            st.session_state.req_text = ex["text"]

user_requirements = st.text_area(
    "Requirements Input",
    value=st.session_state.req_text,
    placeholder="Describe technical scope or click an example above...",
    height=150, label_visibility="collapsed",
)
st.session_state.req_text = user_requirements

selected_epic_key = None
new_epic_title    = ""
if jira_mode and assign_to_epic and project_key and jira_server:
    try:
        jira  = connect_jira(jira_server, jira_email, jira_token)
        epics = jira.search_issues(f'project="{project_key}" AND issuetype=Epic ORDER BY created DESC', maxResults=10)
        opts  = {f"{e.key}: {e.fields.summary}": e.key for e in epics}
        opts["➕ Create New Epic"] = "CREATE_NEW"
        choice = st.selectbox("Target Epic", list(opts.keys()))
        selected_epic_key = opts[choice]
        if selected_epic_key == "CREATE_NEW":
            new_epic_title = st.text_input("New Epic Title")
    except: selected_epic_key = "CREATE_NEW"

# ==========================================
# PIPELINE
# ==========================================
if st.button("🚀 Run AI Agentic Pipeline", use_container_width=True):
    if user_requirements:
        try:
            progress_bar = st.progress(0, text="Starting pipeline...")
            max_iter  = 2
            iteration = 0
            passed    = False
            pm_output = user_requirements
            tech_tasks, feedback, culprit, risk_context = [], "", "NONE", ""

            while iteration < max_iter and not passed:
                iteration += 1

                if create_user_story and (iteration == 1 or culprit == "PM"):
                    progress_bar.progress(15, text="Processing...")
                    pm_res    = pm_chain.invoke({
                        "feedback_context": f"QA Feedback: {feedback}" if culprit == "PM" else "",
                        "requirements": user_requirements,
                        "global_ac": global_ac_text or "None"
                    })
                    pm_output = "\n".join([f"STORY: {s.summary} | AC: {s.ac}" for s in pm_res.stories])

                if iteration == 1:
                    progress_bar.progress(35, text="Processing...")
                    risk_res = risk_chain.invoke({"requirements": user_requirements, "stories": pm_output})
                    st.session_state.risk_result = risk_res.model_dump()
                    risk_context = f"Overall: {risk_res.overall_risk_level}. " + \
                                   " | ".join([f"{r.severity}: {r.title}" for r in risk_res.risks])

                progress_bar.progress(55, text="Processing...")
                arch_res   = arch_chain.invoke({
                    "feedback_context": f"QA Feedback: {feedback}" if culprit == "ARCHITECT" else "",
                    "stories": pm_output, "risk_context": risk_context
                })
                tech_tasks = [t.model_dump() for t in arch_res.tasks]

                progress_bar.progress(75, text="Processing...")
                audit_res  = audit_chain.invoke({
                    "req": user_requirements, "pm_output": pm_output, "tech_plan": str(tech_tasks)
                })

                if audit_res.status == "PASS":
                    passed = True
                else:
                    culprit, feedback = audit_res.culprit, audit_res.report

            progress_bar.progress(90, text="Processing...")
            tasks_str  = "\n".join([f"- [{t['priority']}] {t['summary']} ({t['sp']} SP)" for t in tech_tasks])
            sprint_res = sprint_chain.invoke({
                "velocity": velocity, "sprint_days": sprint_days,
                "start_date": start_date.isoformat(), "tasks": tasks_str
            })
            st.session_state.sprint_plan     = sprint_res.model_dump()
            st.session_state.workflow_result = tech_tasks
            progress_bar.progress(100, text="Done!")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Enter requirements first.")

# ==========================================
# RISK REPORT
# ==========================================
if st.session_state.risk_result:
    rd      = st.session_state.risk_result
    overall = rd.get("overall_risk_level", "Unknown")
    icon    = SEVERITY_ICON.get(overall, "⚪")

    with st.expander(f"{icon} Risk Analysis — Overall: **{overall}**", expanded=False):
        st.markdown(f"_{rd.get('summary', '')}_")
        st.divider()
        for risk in rd.get("risks", []):
            sev = risk.get("severity", "Medium")
            with st.expander(
                f"{SEVERITY_ICON.get(sev, '⚪')} **{risk.get('title')}** — {sev}",
                expanded=False
            ):
                st.markdown(risk.get("description", ""))
                st.divider()
                st.markdown(f"🛡️ **Mitigation plan:** {risk.get('mitigation', '')}")

# ==========================================
# SPRINT TIMELINE
# ==========================================
if st.session_state.sprint_plan and st.session_state.workflow_result:
    plan         = st.session_state.sprint_plan
    sprint_tasks = plan.get("sprint_tasks", [])
    goals        = plan.get("sprint_goals", [])
    notes        = plan.get("planning_notes", "")
    backlog_sp   = {t["summary"]: t["sp"] for t in st.session_state.workflow_result}

    sprints: dict = {}
    for st_task in sprint_tasks:
        sprints.setdefault(st_task["sprint_number"], []).append(st_task)

    num_sprints = max(sprints.keys()) if sprints else 1
    total_sp    = sum(backlog_sp.values())
    total_days  = num_sprints * sprint_days
    end_date    = start_date + timedelta(days=total_days)

    with st.expander("📅 Sprint Timeline & Gantt", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sprints", num_sprints)
        m2.metric("Total SP",      total_sp)
        m3.metric("Duration",      f"{total_days} days")
        m4.metric("Est. End Date", end_date.strftime("%b %d, %Y"))

        if notes:
            st.info(f"💬 {notes}")

        st.divider()

        gantt_rows = []
        for sn in sorted(sprints.keys()):
            color        = SPRINT_COLORS[(sn - 1) % len(SPRINT_COLORS)]
            sprint_start = start_date + timedelta(days=(sn - 1) * sprint_days)
            sprint_end   = sprint_start + timedelta(days=sprint_days - 1)
            goal         = goals[sn - 1] if sn - 1 < len(goals) else f"Sprint {sn}"
            tasks_in     = sprints[sn]
            sprint_sp    = sum(backlog_sp.get(t["task_summary"], 0) for t in tasks_in)

            cards = ""
            for t in tasks_in:
                deps   = f'<div style="color:#94a3b8;font-size:11px;margin-top:3px;">🔗 {", ".join(t["dependencies"])[:60]}</div>' if t.get("dependencies") else ""
                sp_val = backlog_sp.get(t["task_summary"], 0)
                cards += f'''
                <div style="background:{color}18;border:1.5px solid {color}55;border-radius:6px;
                            padding:6px 10px;font-size:12px;max-width:300px;">
                  <div style="font-weight:600;color:#e2e8f0;">{t["task_summary"][:55]}{"…" if len(t["task_summary"])>55 else ""}</div>
                  <div style="color:#94a3b8;font-size:11px;">{sp_val} SP</div>{deps}
                </div>'''

            gantt_rows.append(f'''
            <div style="margin-bottom:20px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <div style="background:{color};color:white;border-radius:6px;padding:3px 12px;font-weight:700;font-size:13px;">Sprint {sn}</div>
                <div style="font-size:12px;color:#94a3b8;">
                  {sprint_start.strftime('%b %d')} → {sprint_end.strftime('%b %d')}
                  &nbsp;·&nbsp;<b style="color:#e2e8f0;">{sprint_sp} SP</b>&nbsp;·&nbsp;{goal}
                </div>
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:8px;padding-left:4px;">{cards}</div>
            </div>''')

        st.html(f'<div style="font-family:sans-serif;padding:16px;background:#0f172a;border-radius:10px;">{"".join(gantt_rows)}</div>')

        st.divider()
        st.markdown("#### 📊 Sprint Breakdown")

        for sn in sorted(sprints.keys()):
            color        = SPRINT_COLORS[(sn - 1) % len(SPRINT_COLORS)]
            sprint_start = start_date + timedelta(days=(sn - 1) * sprint_days)
            sprint_end   = sprint_start + timedelta(days=sprint_days - 1)
            goal         = goals[sn - 1] if sn - 1 < len(goals) else ""
            tasks_in     = sprints[sn]
            sprint_sp    = sum(backlog_sp.get(t["task_summary"], 0) for t in tasks_in)
            load_pct     = min(int(sprint_sp / velocity * 100), 100) if velocity else 0
            load_color   = "#22c55e" if load_pct <= 80 else "#f97316" if load_pct <= 100 else "#ef4444"

            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**Sprint {sn}** — {goal}")
                    st.caption(f"{sprint_start.strftime('%b %d')} – {sprint_end.strftime('%b %d, %Y')}")
                with c2:
                    st.metric("Story Points", f"{sprint_sp} / {velocity}")
                with c3:
                    st.markdown("**Load**")
                    st.html(f'''
                    <div style="background:#1e293b;border-radius:4px;height:22px;margin-top:4px;overflow:hidden;">
                      <div style="background:{load_color};width:{load_pct}%;height:100%;border-radius:4px;
                                  display:flex;align-items:center;justify-content:center;
                                  color:white;font-size:11px;font-weight:700;">{load_pct}%</div>
                    </div>''')

                for t in tasks_in:
                    sp_val = backlog_sp.get(t["task_summary"], 0)
                    deps   = f" 🔗 _{', '.join(t['dependencies'][:2])}_" if t.get("dependencies") else ""
                    st.markdown(f"- **{t['task_summary']}** ({sp_val} SP){deps}")
                    if t.get("reason"):
                        st.caption(f"  {t['reason']}")

# ==========================================
# BACKLOG REVIEW
# ==========================================
if st.session_state.workflow_result:
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    backlog = sorted(
        st.session_state.workflow_result,
        key=lambda x: priority_order.get(x.get("priority", "Medium"), 2)
    )
    st.session_state.workflow_result = backlog

    sprint_map = {}
    if st.session_state.sprint_plan:
        for st_task in st.session_state.sprint_plan.get("sprint_tasks", []):
            sprint_map[st_task["task_summary"]] = st_task["sprint_number"]

    st.subheader("📋 Backlog — sorted by priority")

    for idx, story in enumerate(backlog):
        sn       = sprint_map.get(story["summary"])
        sp_color = SPRINT_COLORS[(sn - 1) % len(SPRINT_COLORS)] if sn else "#475569"

        with st.container(border=True):
            h1, h2 = st.columns([5, 2])
            with h1:
                st.markdown(f"#### 🛠️ Item #{idx + 1}")
                if story.get("priority_reason"):
                    st.caption(f"💡 {story['priority_reason']}")
            with h2:
                if sn:
                    st.html(f'''
                    <div style="display:flex;justify-content:flex-end;margin-bottom:6px;">
                      <span style="background:{sp_color};color:white;border-radius:6px;
                                   padding:4px 14px;font-size:13px;font-weight:700;">Sprint {sn}</span>
                    </div>''')

            story["summary"]     = st.text_input(f"Summary #{idx}", value=story.get("summary", ""), key=f"s_{idx}")
            story["description"] = st.text_area("Technical Plan", value=story.get("description", ""), height=150, key=f"p_{idx}")

            st.markdown("**Acceptance Criteria (AC)**")
            story["acceptance_criteria"] = st.text_area(
                "AC", value=story.get("acceptance_criteria", ""), height=120,
                key=f"ac_{idx}", label_visibility="collapsed"
            )

            st.markdown("**Sub-tasks**")
            for t_idx, sub in enumerate(story.get("subtasks", [])):
                story["subtasks"][t_idx] = st.text_input(
                    f"Task {idx}.{t_idx}", value=sub,
                    key=f"sub_{idx}_{t_idx}", label_visibility="collapsed"
                )

            st.markdown("**Story Points**")
            sp_val  = story.get("sp", 3)
            sp_html = "".join([
                f'<div style="background:{"#6366f1" if sp == sp_val else "#1e293b"};'
                f'color:{"white" if sp == sp_val else "#94a3b8"};'
                f'border:1.5px solid {"#6366f1" if sp == sp_val else "#334155"};'
                f'border-radius:6px;padding:5px 14px;font-size:14px;font-weight:{"700" if sp == sp_val else "400"};'
                f'display:inline-block;margin-right:6px;">{sp}</div>'
                for sp in FIBONACCI
            ])
            st.html(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 12px 0;">{sp_html}</div>')

            st.markdown("**Priority**")
            selected_priority = st.segmented_control(
                "Priority selector", options=PRIORITIES,
                default=story.get("priority", "Medium"),
                key=f"prseg_{idx}", label_visibility="collapsed"
            )
            if selected_priority:
                story["priority"] = selected_priority

    # ==========================================
    # PUSH TO JIRA
    # ==========================================
    if jira_mode:
        if st.button("🚀 Push to Jira", type="primary", use_container_width=True):
            try:
                jira = connect_jira(jira_server, jira_email, jira_token)
                with st.status("Deploying to Jira...", expanded=True) as deploy_status:

                    # --- Project & issue types ---
                    p_info   = jira.project(project_key)
                    i_types  = {it.name.lower(): it.id for it in p_info.issueTypes}
                    s_type   = i_types.get("story", i_types.get("task"))
                    sub_type = i_types.get("sub-task", i_types.get("subtask"))

                    # --- Epic ---
                    pk = selected_epic_key if selected_epic_key != "CREATE_NEW" else None
                    if selected_epic_key == "CREATE_NEW":
                        pk = jira.create_issue(fields={
                            "project": project_key, "summary": new_epic_title,
                            "issuetype": {"id": i_types.get("epic")}
                        }).key
                        st.write(f"✅ Epic created: {pk}")

                    # --- Auto-create sprints ---
                    jira_sprint_ids = {}  # sprint_number -> real jira sprint id

                    if create_sprints and st.session_state.board_id and st.session_state.sprint_plan:
                        board_id     = int(st.session_state.board_id)
                        plan         = st.session_state.sprint_plan
                        goals        = plan.get("sprint_goals", [])
                        sprint_tasks = plan.get("sprint_tasks", [])

                        # Collect unique sprint numbers
                        sprint_numbers = sorted(set(t["sprint_number"] for t in sprint_tasks))

                        st.write(f"📅 Creating {len(sprint_numbers)} sprint(s) on board {board_id}...")
                        for sn in sprint_numbers:
                            s_start = start_date + timedelta(days=(sn - 1) * sprint_days)
                            s_end   = s_start + timedelta(days=sprint_days - 1)
                            s_goal  = goals[sn - 1] if sn - 1 < len(goals) else f"Sprint {sn}"
                            s_name  = f"Sprint {sn} — {s_goal[:40]}"

                            try:
                                jira_sprint_id = create_sprint(jira, board_id, s_name, s_start, s_end)
                                jira_sprint_ids[sn] = jira_sprint_id
                                st.write(f"  ✅ {s_name} (ID: {jira_sprint_id})")
                            except Exception as e:
                                st.warning(f"  ⚠️ Could not create sprint {sn}: {e}")

                    # --- Create issues ---
                    sprint_issue_map = {}  # sprint_number -> [issue_keys]
                    st.write(f"📝 Creating {len(backlog)} issue(s)...")

                    for s in backlog:
                        desc = f"h3. Acceptance Criteria\n{s.get('acceptance_criteria')}\n\nh3. Technical Plan\n{s.get('description')}"
                        if s.get("priority_reason"):
                            desc += f"\n\nh3. Priority Reasoning\n{s['priority_reason']}"

                        fields = {
                            "project":     project_key,
                            "summary":     f"[AI] {s.get('summary')}",
                            "description": desc,
                            "issuetype":   {"id": s_type},
                            "priority":    {"name": JIRA_PRI_MAP.get(s.get("priority", "Medium"), "Medium")},
                            st.session_state.story_id_value: float(s.get("sp", 3))
                        }
                        if pk: fields["parent"] = {"key": pk}

                        issue = jira.create_issue(fields=fields)

                        # Track issue key per sprint for batch move
                        sn = sprint_map.get(s.get("summary"))
                        if sn:
                            sprint_issue_map.setdefault(sn, []).append(issue.key)

                        # Subtasks
                        for sub in s.get("subtasks", []):
                            jira.create_issue({
                                "project":   project_key,
                                "summary":   sub,
                                "issuetype": {"id": sub_type},
                                "parent":    {"id": issue.id}
                            })

                    # --- Move issues into sprints ---
                    if jira_sprint_ids:
                        st.write("🏃 Assigning issues to sprints...")
                        for sn, issue_keys in sprint_issue_map.items():
                            jira_sprint_id = jira_sprint_ids.get(sn)
                            if jira_sprint_id and issue_keys:
                                try:
                                    move_to_sprint(jira, jira_sprint_id, issue_keys)
                                    st.write(f"  ✅ Sprint {sn}: {len(issue_keys)} issue(s) assigned")
                                except Exception as e:
                                    st.warning(f"  ⚠️ Sprint {sn} assign failed: {e}")

                    deploy_status.update(label="✅ Done!", state="complete")

                st.success(f"Pushed {len(backlog)} issues with {len(jira_sprint_ids)} sprint(s) to Jira!")
                for k in ["workflow_result", "risk_result", "sprint_plan"]:
                    st.session_state[k] = None

            except Exception as e:
                st.error(f"Jira Error: {e}")
    else:
        st.info("✅ Test mode — enable Jira Integration in sidebar to push.")

st.divider()
st.caption("Agentic AI Sprint Architect 2026")