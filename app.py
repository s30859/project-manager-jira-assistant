import os
import streamlit as st
from jira import JIRA
from dotenv import load_dotenv
import time

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()

# Configuration
MODEL_NAME = "gemini-3-pro-preview" 
FIBONACCI = [1, 2, 3, 5, 8, 13]

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def connect_jira(server, email, token):
    return JIRA(server=server, basic_auth=(email, token))

# ==========================================
# 1. DATA SCHEMAS
# ==========================================
class UserStory(BaseModel):
    summary: str = Field(description="User story format: 'As a [user], I want [action] so that [value]'")
    ac: str = Field(description="Strict Acceptance Criteria for this story")

class UserStoryList(BaseModel):
    stories: List[UserStory]

class TechTask(BaseModel):
    summary: str = Field(description="Exact original summary from PM story")
    description: str = Field(description="Step-by-step technical implementation plan")
    acceptance_criteria: str = Field(description="Copy the EXACT Acceptance Criteria from the PM story")
    sp: int = Field(description="Fibonacci estimation", default=3)
    subtasks: List[str] = Field(description="List of actionable technical subtasks")

class TechTaskList(BaseModel):
    tasks: List[TechTask]

class AuditResult(BaseModel):
    status: Literal["PASS", "FAIL"]
    culprit: Literal["PM", "ARCHITECT", "NONE"]
    report: str

# ==========================================
# 2. AI PIPELINES
# ==========================================
pm_prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as a Senior Business Analyst. Convert requirements into User Stories. "
               "If Global AC is 'None', you MUST generate 3-5 specific AC for EACH story yourself. "
               "{feedback_context}"),
    ("human", "Requirements: {requirements}\nGlobal Acceptance Criteria: {global_ac}")
])
pm_chain = pm_prompt | llm.with_structured_output(UserStoryList)

arch_prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as a Lead Architect. Decompose stories into technical tasks. "
               "MANDATORY: Preserve the EXACT 'ac' from the PM for each story "
               "and put it into the 'acceptance_criteria' field. DO NOT merge it into the description. "
               "{feedback_context}"),
    ("human", "Input stories from PM: {stories}")
])
arch_chain = arch_prompt | llm.with_structured_output(TechTaskList)

audit_prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as QA Auditor. Compare requirements with the technical plan. "
               "Ensure 'acceptance_criteria' field is present and matches PM's intent. "
               "FAIL if the Architect ignored or lost the specific AC."),
    ("human", "Original Req: {req}\nPM Stories: {pm_output}\nTech Plan: {tech_plan}")
])
audit_chain = audit_prompt | llm.with_structured_output(AuditResult)

# ==========================================
# 3. UI SETUP (Streamlit Sidebar & Config)
# ==========================================
st.set_page_config(page_title="Strategic Requirements Decomposition", layout="wide")

if 'workflow_result' not in st.session_state:
    st.session_state.workflow_result = None
if 'story_id_value' not in st.session_state:
    st.session_state.story_id_value = "customfield_10016"

with st.sidebar:
    st.header("🔑 Jira Authentication")
    jira_server = st.text_input("Jira Server URL", placeholder="https://your-domain.atlassian.net")
    jira_email = st.text_input("Jira Email")
    jira_token = st.text_input("Jira API Token", type="password")
    
    st.divider()
    project_key = st.text_input("Project key", placeholder="e.g., SCRUM")
    
    with st.expander("🛠️ Advanced Settings"):
        if st.button("Detect Story Points ID", use_container_width=True):
            if jira_server and jira_email and jira_token:
                try:
                    temp_jira = connect_jira(jira_server, jira_email, jira_token)
                    sp_f = [f for f in temp_jira.fields() if 'story point' in f['name'].lower()]
                    if sp_f:
                        st.session_state.story_id_value = sp_f[0]['id']
                        st.success(f"Detected: {sp_f[0]['id']}")
                except: st.error("Detection failed.")
        st.text_input("Story ID Field", value=st.session_state.story_id_value, disabled=True)

    st.divider()
    st.subheader("Configuration")
    
    # ВСЕ ЧЕКБОКСЫ ВЕРНУЛИСЬ:
    use_global_ac = st.checkbox("Add Custom Acceptance Criteria", value=False)
    global_ac_text = st.text_area("Criteria", placeholder="e.g. Test coverage > 80%...") if use_global_ac else ""
    
    assign_to_sprint = st.checkbox("Assign to one sprint", value=False)
    sprint_id = st.text_input("Sprint ID", placeholder="e.g. 10") if assign_to_sprint else ""
    
    assign_to_epic = st.checkbox("Assign to Epic", value=True)
    create_user_story = st.checkbox("Create User Story", value=True)
    
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.workflow_result = None
        st.rerun()

st.title("Strategic Requirements Decomposition")

user_requirements = st.text_area("Requirements Input", placeholder="Describe technical scope...", height=150, label_visibility="collapsed")

selected_epic_key = None
new_epic_title = ""
if assign_to_epic and project_key and jira_server:
    try:
        jira = connect_jira(jira_server, jira_email, jira_token)
        existing_epics = jira.search_issues(f'project = "{project_key}" AND issuetype = Epic ORDER BY created DESC', maxResults=10)
        epic_options = {f"{e.key}: {e.fields.summary}": e.key for e in existing_epics}
        epic_options["➕ Create New Epic"] = "CREATE_NEW"
        choice = st.selectbox("Select Target Epic", options=list(epic_options.keys()))
        selected_epic_key = epic_options[choice]
        if selected_epic_key == "CREATE_NEW":
            new_epic_title = st.text_input("New Epic Title")
    except: selected_epic_key = "CREATE_NEW"

# ==========================================
# 4. AGENTIC WORKFLOW
# ==========================================
if st.button("🚀 Run AI Agentic Pipeline", use_container_width=True):
    if user_requirements and project_key:
        try:
            with st.status("Agents are working...", expanded=True) as status:
                max_iterations = 2
                iteration = 0
                workflow_passed = False
                pm_output = user_requirements
                tech_tasks = []
                feedback = ""
                culprit = "NONE"

                while iteration < max_iterations and not workflow_passed:
                    iteration += 1
                    st.write(f"### Iteration {iteration}")

                    if create_user_story and (iteration == 1 or culprit == "PM"):
                        pm_res = pm_chain.invoke({
                            "feedback_context": f"QA Feedback: {feedback}" if culprit == "PM" else "",
                            "requirements": user_requirements,
                            "global_ac": global_ac_text if global_ac_text else "None"
                        })
                        pm_output = "\n".join([f"STORY: {s.summary} | AC: {s.ac}" for s in pm_res.stories])

                    arch_res = arch_chain.invoke({
                        "feedback_context": f"QA Feedback: {feedback}" if culprit == "ARCHITECT" else "",
                        "stories": pm_output
                    })
                    tech_tasks = [task.model_dump() for task in arch_res.tasks]

                    audit_res = audit_chain.invoke({
                        "req": user_requirements,
                        "pm_output": pm_output,
                        "tech_plan": str(tech_tasks)
                    })
                    
                    if audit_res.status == "PASS": workflow_passed = True
                    else:
                        culprit, feedback = audit_res.culprit, audit_res.report
                        st.warning(f"Reject by QA: {culprit}")

                st.session_state.workflow_result = tech_tasks
                status.update(label="Complete!", state="complete")
        except Exception as e: st.error(f"Error: {e}")

# ==========================================
# 5. REVIEW & DEPLOY (Vertical Stack Order)
# ==========================================
if st.session_state.workflow_result:
    backlog = st.session_state.workflow_result
    
    for idx, story in enumerate(backlog):
        with st.container(border=True):
            st.markdown(f"#### 🛠️ Item #{idx + 1}")
            
            # 1. Summary
            story['summary'] = st.text_input("Summary", value=story.get('summary', ''), key=f"s_{idx}")
            
            # 2. Sub-tasks
            st.markdown("**Sub-tasks**")
            subs = story.get('subtasks', [])
            for t_idx, sub in enumerate(subs):
                story['subtasks'][t_idx] = st.text_input(f"Task {t_idx}", value=sub, key=f"sub_{idx}_{t_idx}", label_visibility="collapsed")
            
            # 3. Story Points
            current_sp = story.get('sp', 3)
            story['sp'] = st.segmented_control("Complexity (Points)", options=FIBONACCI, default=current_sp if current_sp in FIBONACCI else 3, key=f"sp_{idx}")
            
            # 4. Technical Implementation Plan
            st.markdown("**Technical Implementation Plan**")
            story['description'] = st.text_area("Tech Plan Content", value=story.get('description', ''), height=150, key=f"p_{idx}", label_visibility="collapsed")
            
            # 5. Acceptance Criteria (В САМОМ КОНЦЕ ПОД TECH PLAN)
            st.markdown("**Acceptance Criteria (AC)**")
            current_ac = story.get('acceptance_criteria', story.get('ac', 'No AC generated'))
            story['acceptance_criteria'] = st.text_area("AC Content", value=current_ac, height=150, key=f"ac_{idx}", label_visibility="collapsed")

    if st.button("🚀 Push Verified Plan to Jira", type="primary", use_container_width=True):
        try:
            jira = connect_jira(jira_server, jira_email, jira_token)
            with st.status("Deploying..."):
                p_info = jira.project(project_key)
                i_types = {it.name.lower(): it.id for it in p_info.issueTypes}
                s_type = i_types.get('story', i_types.get('task'))
                sub_type = i_types.get('sub-task', i_types.get('subtask'))

                pk = selected_epic_key if selected_epic_key != "CREATE_NEW" else None
                if selected_epic_key == "CREATE_NEW":
                    e_dict = {'project': project_key, 'summary': new_epic_title, 'issuetype': {'id': i_types.get('epic')}}
                    pk = jira.create_issue(fields=e_dict).key

                for s in backlog:
                    # Комбинируем для Jira
                    jira_desc = f"h3. Technical Plan\n{s.get('description')}\n\nh3. Acceptance Criteria\n{s.get('acceptance_criteria')}"
                    
                    fields = {
                        'project': project_key, 'summary': f"[AI] {s.get('summary')}",
                        'description': jira_desc, 'issuetype': {'id': s_type},
                        st.session_state.story_id_value: float(s.get('sp', 3))
                    }
                    if pk: fields['parent'] = {'key': pk}
                    if assign_to_sprint and sprint_id: fields['customfield_10020'] = int(sprint_id)
                    
                    issue = jira.create_issue(fields=fields)
                    for sub in s.get('subtasks', []):
                        jira.create_issue({'project': project_key, 'summary': sub, 'issuetype': {'id': sub_type}, 'parent': {'id': issue.id}})
            st.success("Done!")
            st.session_state.workflow_result = None
        except Exception as e: st.error(f"Jira Error: {e}")

st.divider()
st.caption("Agentic AI Sprint Architect 2026")